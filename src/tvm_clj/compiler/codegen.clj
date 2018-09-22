(ns tvm-clj.compiler.codegen
  (:require [tvm-clj.core :as c]
            [tvm-clj.api :as api]
            [tvm-clj.compiler.ast :as ast]
            [tvm-clj.compiler.graph :as g]
            [tvm-clj.compute.tensor-math :as tm]
            [tvm-clj.compute.registry :as tvm-reg]
            [tech.datatype.base :as dtype]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.utils :as tens-utils]
            [tech.compute.tensor.dimensions :refer [when-not-error] :as ct-dims]
            [clojure.core.matrix.protocols :as mp]
            [clojure.set :as c-set]))


(defn input-fn->graph
  [input-fn graph & input-fn-args]
  (let [*graph (atom graph)
        compile-stream (ast/compile-stream *graph)]
    (ct/with-stream compile-stream
      (apply input-fn input-fn-args))
    @*graph))


(defmulti edge-metadata
  "Return a map of metadata that describes an edge"
  :type)


(defmethod edge-metadata :select [edge] {:location :host})
(defmethod edge-metadata :transpose [edge] {:location :host})
(defmethod edge-metadata :static-cast [edge] {:location :device
                                              :operation :element-wise})
(defmethod edge-metadata :binary-op [edge] {:location :device
                                            :operation :element-wise})


(defn graph->host-device-graphs
  "Given a graph, split it into host and device graphs"
  [graph]
  (let [location-detector (comp #{:host} :location edge-metadata)
        host-edges (->> (g/edges graph)
                        (filter location-detector))
        device-edges (->> (g/edges graph)
                          (remove location-detector))]
    {:host-graph (assoc graph :edges host-edges)
     :device-graph (assoc graph :edges device-edges)}))


(defn can-combine-edge?
  [prev-edge next-edge]
  (and (= (:output prev-edge)
          (:input next-edge))
       (= (:operation (edge-metadata prev-edge))
          (:operation (edge-metadata next-edge)))))


(defn partition-device-graph-into-operations
  "Given a device graph partition the graph into different operations;
operations can be reductive, element-wise, or scanning.
Operations can be combined if the have data dependencies and operate over
the same result bounds."
  [device-graph]
  (let [edge-lists (->> (:edges device-graph)
                        (reduce (fn [edge-lists next-edge]
                                  (if-let [valid-edge-list-idx (->> (map-indexed vector edge-lists)
                                                                    (filter #(can-combine-edge? (last (second %)) next-edge))
                                                                    ffirst)]
                                    (assoc edge-lists valid-edge-list-idx (conj (get edge-lists valid-edge-list-idx) next-edge))
                                    (conj edge-lists [next-edge])))
                                []))]
    (mapv #(assoc device-graph :edges %) edge-lists)))


(defn graph->read-operations
  [operation-graph]
  (let [leaf (g/get-tensor operation-graph (first (g/leaves operation-graph)))
        n-dims (count (mp/get-shape leaf))
        roots (mapv #(g/get-tensor operation-graph %) (g/roots operation-graph))]
    ;;Define if this is a custom read operation or if it is a tvm read operation
    (mapv (fn [root-node]
            (let [node-shape (get-in root-node [:dimensions :shape])]
              {:node-id (:id root-node)
               :n-dims n-dims
               :shape node-shape
               :node root-node
               :read-type
               (if (every? (fn [shape-item]
                             (cond
                               (keyword? shape-item)
                               true
                               (number? shape-item)
                               true
                               (sequential? shape-item)
                               (ct-dims/monotonically-increasing? shape-item)))
                           node-shape)
                 :tvm-read
                 :custom-read)}))
          roots)))


(defn graph-seq->read-operations
  [op-graphs]
  (->> op-graphs
       (mapcat graph->read-operations)
       set))

(defn create-n-vars
  [n stem dtype]
  (->> (range n)
       (mapv (fn [idx]
               (api/variable (str stem "_" idx) :dtype "int32")))))


(defn- lookup-idx
  "Lookup the index for this dimension"
  [index-var shape-var stride-var shape-entry]
  (let [index-var
        (if (and (sequential? shape-entry)
                 ;;If the sequence *is* monotonically increasing then
                 ;;whatever select produced them will be valid.  The only
                 ;;time we have a compile time mapping is if the sequence
                 ;;is *not* monotonically increasing.
                 (not (ct-dims/monotonically-increasing? shape-entry)))
          ;;ugghhh.  Build a large if-else lookup table and hope the
          ;;compiler can build a decent lookup table.
          (->> (map-indexed vector shape-entry)
               (reduce (fn [table [idx number]]
                         (let [bool-stmt (api/eq index-var (api/const (long idx)
                                                                      :dtype "int32"))
                               true-stmt (api/const (long number) :dtype "int32")
                               false-stmt table]
                           (api/select bool-stmt true-stmt false-stmt)))
                       index-var))
          index-var)]
    (api/mul stride-var (api/mod index-var shape-var))))


(defn custom-read-operation
  "A custom read bindings when the read operation falls outside the scope
  of what tvm supports."
  [index-vars tvm-tensor shape-vars stride-vars compile-time-shape]
  (api/tget
   tvm-tensor
   [(->> (map lookup-idx
              index-vars shape-vars stride-vars compile-time-shape)
         (reduce api/add))]))


(defn read-var
  [var res-dtype variable-map index-vars]
  (if (number? var)
    (api/const (tens-utils/dtype-cast var res-dtype) :dtype res-dtype)
    (let [read-op (get-in variable-map [(:id var) :read-operation])]
      (read-op index-vars))))


(defmulti tvm-eltwise-binding
  "Perform the specific tvm eltwise operation.  Return an updated variable map with the
  :read-operation updated with a function that takes the index vars and produces a value
  for the approprate variable (result of the edge operation.

Returns a new variable map.

Dispatch on edge type."
  (fn [variable-map graph index-vars edge]
    (:type edge)))


(defmethod tvm-eltwise-binding :static-cast
  [variable-map graph index-vars edge]
  (let [[item dtype _] (:args edge)]
    (api/static-cast (name dtype) (read-var item dtype variable-map index-vars))))


(defmethod tvm-eltwise-binding :binary-op
  [variable-map graph index-vars edge]
  (let [[lhs rhs op _] (:args edge)
        op-dtype (-> (g/get-tensor graph (first (:output edge)))
                     dtype/get-datatype)
        tvm-op (condp = op
                 :+ api/add
                 :- api/sub
                 :rem api/mod
                 :/ api/div
                 :* api/mul)]
    (->> [lhs rhs]
         (map #(read-var % op-dtype variable-map index-vars))
         (apply tvm-op)
         ;;Keep the types the same; tvm does widening operations
         (api/static-cast (name op-dtype)))))


(defmulti read-op->variable-map-entry
  "Produce the necessary read operation and binding information
  for the roots of the graph."
  (fn [read-op & args]
    (:read-type read-op)))


(defn- safe-node-name
  [node]
  (api/safe-str (name (:id node))))


(defmethod read-op->variable-map-entry :custom-read
  [read-op n-dims operation-graph]
  (let [node (:node read-op)
        node-buffer (ast/get-buffer operation-graph (:bufname node))
        datatype (dtype/get-datatype node)
        dtype-name (name datatype)
        node-name (safe-node-name node)
        buffer-ecount (api/variable (str node-name "_buffer_ecount") :dtype "int32")
        placeholder (api/placeholder [buffer-ecount]
                                     :dtype dtype-name :name (str node-name "_buffer"))
        shape-vars (create-n-vars n-dims (str node-name "_shape") "int32")
        stride-vars (create-n-vars n-dims (str node-name "_stride") "int32")
        elem-offset (if (:byte-offset? node-buffer)
                      (api/variable (str node-name "_elem_offset") :dtype "int32")
                      0)]
    (assoc read-op
           :tensor placeholder
           :shape-vars shape-vars
           :stride-vars stride-vars
           :bind-buffer (api/declare-buffer [buffer-ecount]
                                            :dtype dtype-name
                                            :elem-offset elem-offset)
           ;;What variables need to be bound in the declaration
           :declaration-bind-list (concat [placeholder] shape-vars stride-vars)
           ;;Function to produce the binding at the callsite...This produces a list that needs to be in the
           ;;same order as the above list
           :callsite-bind-list-fn (fn [tensor]
                                    (let [tens-shape (tm/left-pad-ones
                                                      (mp/get-shape tensor)
                                                      n-dims)
                                          tens-stride (ct-dims/extend-strides
                                                       tens-shape
                                                       (get-in tensor [:dimensions :strides]))]
                                      ;;We bind the raw buffer as a one-dimensional buffer.
                                      (concat [(ct/tensor->buffer tensor)]
                                              (map int tens-shape)
                                              (map int tens-stride))))
           :read-operation #(custom-read-operation % placeholder
                                                   shape-vars stride-vars
                                                   (get-in node [:dimensions :shape])))))


(defmethod read-op->variable-map-entry :tvm-read
  [read-op n-dims operation-graph]
  (let [node (:node read-op)
        node-buffer (ast/get-buffer operation-graph (:bufname node))
        node-name (safe-node-name node)
        datatype (dtype/get-datatype node)
        dtype-name (name datatype)
        shape-vars (create-n-vars n-dims (str node-name "_shape") "int32")
        placeholder (api/placeholder shape-vars
                                     :dtype dtype-name
                                     :name node-name)
        elem-offset (if (:byte-offset? node-buffer)
                      (api/variable (str node-name "_elem_offset") :dtype "int32")
                      0)]
    (assoc read-op
           :tensor placeholder
           :bind-buffer (api/declare-buffer shape-vars
                                            :dtype dtype-name
                                            :strides (when (:sparse? node)
                                                       (create-n-vars n-dims (str node-name "_stride") "int32"))
                                            :elem-offset elem-offset)
           :declaration-bind-list [placeholder]
           :callsite-bind-list-fn (fn [tensor]
                                    ;;Extend the shape to be compatible with the function
                                    (let [tens-shape (tm/left-pad-ones
                                                      (mp/get-shape tensor)
                                                      n-dims)
                                          tens-stride (ct-dims/extend-strides
                                                       tens-shape
                                                       (get-in tensor [:dimensions :shape]))]
                                      [(-> tensor
                                           (assoc-in [:dimensions :shape] tens-shape)
                                           (assoc-in [:dimensions :stride] tens-stride)
                                           (assoc-in [:buffer :byte-offset?] (:byte-offset? node-buffer))
                                           (assoc :sparse? (:sparse? node)))]))
                                 :read-operation #(api/tget placeholder %))))



(defn create-tvm-operation
  "Create and compile a tvm operation."
  [operation-graph]
  (when-not-error (= 1 (count (g/leaves operation-graph)))
    "Operation graph cannot handle multiple outputs"
    {:leaves (g/leaves operation-graph)})
  ;;The leaf defines the iteration variables
  (let [leaf (g/get-tensor operation-graph (first (g/leaves operation-graph)))
        n-dims (count (mp/get-shape leaf))
        variable-map (->> (graph->read-operations operation-graph)
                          (map #(read-op->variable-map-entry % n-dims operation-graph))
                          ;;create a map that finds the bind buffer entry
                          (map (juxt :node-id identity))
                          (into {}))
        last-edge (last (:edges operation-graph))
        output-id (:id leaf)
        _ (when-not-error (= output-id (first (:output last-edge)))
            "Graph leaf does not correspond to the last edge"
            {:leaf-id (:id leaf)
             :output-id (first (:output last-edge))})
        ;;For now we only support injective operations.  But later we will need to
        ;;figure out how to work with a wider range of operations
        ;;(reduce, pooling, scanning)
        output-name (safe-str (name output-id))
        output-shape (create-n-vars n-dims (str output-name "_shape") "int32")
        compute-op (api/compute
                    output-shape
                    (tm/y-dim-tvm-fn
                     n-dims
                     (fn [index-vars]
                       (let [variable-map (reduce (fn [variable-map edge]
                                                    (let [output-id (first (:output edge))]
                                                      (assoc-in variable-map [output-id :read-operation]
                                                                (constantly
                                                                 (tvm-eltwise-binding variable-map operation-graph
                                                                                      index-vars edge)))))
                                                  variable-map
                                                  (:edges operation-graph))
                             last-output-op (get-in variable-map [output-id :read-operation])]
                         (last-output-op index-vars))))
                    :name output-name)
        output-node (g/get-tensor operation-graph output-id)
        output-node-buffer (ast/get-buffer operation-graph (:bufname output-node))
        output-tensor (first (api/output-tensors compute-op))
        variable-map (assoc variable-map output-id
                            {:tensor output-tensor
                             :node-id output-id
                             :bind-buffer (api/declare-buffer output-shape
                                                              :dtype (:dtype output-tensor))
                             :declaration-bind-list [output-tensor]
                             :callsite-bind-list-fn #(vector %)})
        var-map-values (vals variable-map)]
    ;;Return a summarized operation that was described by the graph.
    (assoc operation-graph
           :operation {:type :injective
                       :operation compute-op
                       :bind-map (->> var-map-values
                                      (map #(vector (:tensor %) (:bind-buffer %)))
                                      (into {}))
                       :declaration-bind-list (vec (mapcat :declaration-bind-list var-map-values))
                       :callsite-bind-list-fn (fn [runtime-var-map]
                                                (->> var-map-values
                                                     (mapcat (fn [{:keys [node-id callsite-bind-list-fn]}]
                                                               (if-let [var-map-tensor (get runtime-var-map node-id)]
                                                                 (callsite-bind-list-fn var-map-tensor)
                                                                 (throw (ex-info "Failed to find runtime tensor in arg map"
                                                                                 {:tensor-id node-id
                                                                                  :runtime-ids (keys runtime-var-map)})))))
                                                     vec))})))


(defn- map-edge-args-to-runtime
  [arg-map edge]
  (->> (:args edge)
       (map #(if (ast/is-tensor? %)
               (get arg-map (:id %))
               %))))


(defmulti host-perform-edge
  "Perform the edge function on the host.  Return an updated arg-map."
  (fn [arg-map graph edge & args]
    (:type edge)))


(defmethod host-perform-edge :select
  [arg-map graph edge & args]
  (let [[tensor select-args] (map-edge-args-to-runtime arg-map edge)
        ;;remap the select args to what the runtime system is capable of
        ;;Which means replace non-monotonic sequences with :all
        select-args (->> select-args
                         (map (fn [select-arg]
                                (if (and (sequential? select-arg)
                                         (not (ct-dims/monotonically-increasing?
                                               select-arg)))
                                  :all
                                  select-arg))))
        out-id (first (:output edge))]
    (apply ct/select tensor select-args)))


(defmethod host-perform-edge :transpose
  [arg-map graph edge & args]
  (let [[tensor reorder-vec] (map-edge-args-to-runtime arg-map edge)]
    (ct/transpose tensor reorder-vec)))


(defn compile-fn
  "Produce a compiled function that has the actual function to be called
and a description of the arguments to the function."
  [driver initial-graph input-fn & args]
  (let [fn-graph (apply input-fn->graph input-fn initial-graph args)
        {:keys [host-graph device-graph]} (graph->host-device-graphs fn-graph)
        device-op-graphs (->> (partition-device-graph-into-operations device-graph)
                              (map-indexed vector))
        roots-set (set (g/roots fn-graph))
        leaves-set (set (g/leaves fn-graph))
        total-input-list (c-set/union
                          roots-set
                          leaves-set
                          ;;Because select,transpose,reshap happen on the host,
                          ;;if there is a select operation
                          ;;on a piece of data that data buffer has to 'pop' out
                          ;;of the device graphs.
                          ;;Else the device graphs can communicate tensor buffers
                          ;;to each other.
                          (set (filter (set (g/roots host-graph))
                                       (mapcat g/leaves device-op-graphs))))
        device-op-graphs
        (->> device-op-graphs
             (map (fn [[idx device-op-graph]]
                    (let [compiled-graph (create-tvm-operation device-op-graph)
                          operation (:operation compiled-graph)
                          op-schedule (condp = (:type operation)
                                        :injective (tvm-reg/schedule-injective
                                                    driver
                                                    (:operation operation)))]
                      [idx
                       (assoc-in compiled-graph [:operation :operation]
                                 (api/schedule->lowered-function
                                  op-schedule
                                  (:declaration-bind-list operation)
                                  api/default-build-config
                                  :name (str "fn_" idx)
                                  :bind-map (:bind-map operation)))]))))
        compiled-functions (map #(get-in % [1 :operation :operation]) device-op-graphs)
        module (tvm-reg/->module driver compiled-functions)
        call-functions
        (->> device-op-graphs
             (mapv (fn [[idx device-op-graph]]
                     (assoc device-op-graph
                            :operation
                            {:fn (c/get-module-function module (str "fn_" idx))
                             :callsite-bind-list-fn
                             (get-in device-op-graph
                                     [:operation :callsite-bind-list-fn])}))))
        final-fn (fn [arg-map]
                   (when-not-error (every? #(contains? arg-map %) total-input-list)
                     "Argmap does not include required argument"
                     {:argmap-keys (keys arg-map)
                      :required-args total-input-list})
                   (let [dev-argmap
                         (->> (:edges host-graph)
                              (reduce (fn [arg-map edge]
                                        (assoc arg-map (first (:output edge))
                                               (host-perform-edge arg-map host-graph
                                                                  edge)))
                                      arg-map))]
                     (->> call-functions
                          (map (fn [op-graph]
                                 (let [tvm-fn (get-in op-graph [:operation :fn])
                                       callsite-binders
                                       (get-in op-graph [:operation
                                                         :callsite-bind-list-fn])]
                                   (apply c/call-function tvm-fn (callsite-binders
                                                                  dev-argmap)))))
                          dorun)
                     nil))
        id->arg-definition-fn (fn [id-list]
                                (mapv #(let [tens (g/get-tensor fn-graph %)]
                                         {:id %
                                          :datatype (dtype/get-datatype tens)
                                          :shape (mp/get-shape tens)})
                                      id-list))]
    {:inputs (id->arg-definition-fn roots-set)
     :outputs (id->arg-definition-fn leaves-set)
     :intermediates (id->arg-definition-fn (c-set/difference total-input-list
                                                             roots-set leaves-set))
     :fn! final-fn}))
