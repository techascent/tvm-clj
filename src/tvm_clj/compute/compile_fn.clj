(ns tvm-clj.compute.compile-fn
  (:require [tvm-clj.core :as c]
            [tvm-clj.api :as api]
            [tvm-clj.base :as b]
            [tvm-clj.compute.registry :as tvm-reg]
            [tvm-clj.compute.tensor.functional-protocols :as fnp]
            [tech.compute.driver :as drv]
            [tech.compute.tensor.dimensions :refer [when-not-error] :as ct-dims]
            [tech.datatype.base :as dtype]
            [clojure.core.matrix.protocols :as mp]
            [tech.compute.tensor :as ct]
            [clojure.set :as c-set]
            [tvm-clj.compute.tensor-math :as tm]
            [clojure.string :as c-str]
            [tech.compute.tensor.utils :as tens-utils]))


(defn compute-graph
  []
  {:nodes {}
   :edges []})


(defn- edges
  [graph]
  (get graph :edges))


(defn- edges->map
  [graph key-fn val-fn]
  (->> (edges graph)
       (mapcat (fn [edge]
                 (for [in (:input edge)
                       out (:output edge)]
                   [in out])))
       (group-by key-fn)
       (map (fn [[k v]]
              [k (map val-fn v)]))
       (into {})))


(defn parent->child-map
  [graph]
  (edges->map graph first second))


(defn child->parent-map
  [graph]
  (edges->map graph second first))


(defn roots
  [graph]
  (let [p->c (parent->child-map graph)]
    (c-set/difference (set (keys p->c))
                      (set (apply concat (vals p->c))))))


(defn leaves
  [graph]
  (let [c->p (child->parent-map graph)]
    (c-set/difference (set (keys c->p))
                      (set (apply concat (vals c->p))))))


(defn get-node
  [graph node-id]
  (let [retval (get-in graph [:nodes node-id])]
    (when-not-error retval
      "Failed to find node:"
      {:node-id node-id
       :nodes (keys (get graph :nodes))})
    retval))


(defn generate-id
  [id-stem id-set]
  (loop [idx 1]
    (let [new-id (-> (format "%s-%s" id-stem idx)
                     keyword)]
      (if (contains? id-set new-id)
        (recur (inc idx))
        new-id))))


(defn generate-derived-node-id
  "Generate a derived node id and return the node"
  [node graph]
  (let [node-id (name (:id node))
        num-idx (.lastIndexOf node-id "-")
        ^String id-stem (if (not= -1 num-idx)
                          (.substring node-id 0 num-idx)
                          node-id)
        graph-nodes (:nodes graph)]
    (loop [idx 1]
      (let [new-id (-> (format "%s-%s" id-stem idx)
                       keyword)]
        (if (contains? graph-nodes new-id)
          (recur (inc idx))
          new-id)))))


(defn- get-or-create-node-id
  "Generate an id for this node."
  [graph node]
  (when-not-error (or (nil? (:id node))
                              (nil? (get-in graph [:nodes (:id node)])))
    "Node has ID and it is already in the graph"
    {:new-node node
     :existing-node (get-node graph (:id node))})
  (assoc node :id (generate-id (name (get node :type))
                               (set (keys (get graph :nodes))))))


(defn contains-id?
  [graph id]
  (get-in graph [:nodes id]))


(defn- add-node
  [graph node]
  (when-not-error (not (contains-id? graph (:id node)))
    "Graph contains node."
    {:node-ids (map :id (:nodes graph))
     :node node})
  (assoc-in graph [:nodes (:id node)] node))


(defn make-variable
  "Create a scalar variable.  Returns new graph; error if varname is not unique."
  [graph varname & {:keys [dtype]
                    :or {dtype :int32}}]
  (add-node graph {:type :variable
                   :dtype dtype
                   :id varname}))


(defn variable?
  [graph id]
  (= :variable (get-in graph [:nodes :id :type])))


(defn get-variable
  [graph id]
  (when-not-error (variable? graph id)
    "Node is not variable"
    {:node (get-node graph id)
     :id id})
  (get-node graph id))


(defn typed-variable?
  [graph id dtype]
  (and (variable? graph id)
       (= dtype (get-in graph [:nodes id :dtype]))))


(defn sequence-of-numbers?
  [item-seq]
  (and (sequential? item-seq)
       (every? number? item-seq)))


(defn make-buffer
  [graph bufname & {:keys [dtype byte-offset?]
                    :or {dtype :float32
                         byte-offset? false}}]
  (add-node graph {:type :buffer
                   :dtype dtype
                   :byte-offset? byte-offset?
                   :id bufname}))

(defn get-buffer
  [graph id]
  (let [node (get-node graph id)]
    (when-not-error (= :buffer (:type node))
      "Node is not buffer type"
      {:node node
       :id id})
    node))


(defn- ensure-valid-tensor-shape
  "Shapes are sequences that can be composed of:
- keyword - which must map to an int32 variable.
- sequences of numbers
- a number (which implies (range number))"
  [graph shape]
  (let [invalid-shape-items (->> (remove #(or (number? %)
                                              (typed-variable? graph % :int32)
                                              (sequence-of-numbers? %))
                                         shape)
                                 seq)]
    (when-not-error invalid-shape-items
      "Shape contains invalid entries"
      {:shape shape
       :invalid-items invalid-shape-items})
    shape))


(defprotocol PGraphItemInfo
  (is-tensor? [item]))


(extend-protocol PGraphItemInfo
  Object
  (is-tensor? [item] false))


(defrecord CompileTensor [type id dimensions bufname dtype sparse?]
  dtype/PDatatype
  (get-datatype [_] dtype)
  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (->> (:shape dimensions)
                      (map (fn [shape-item]
                             (if (sequential? shape-item)
                               (count shape-item)
                               shape-item)))
                      (remove #(and (sequential? %)
                                    (= 1 (count %))))
                      vec))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))
  PGraphItemInfo
  (is-tensor? [_] true))


(defn get-tensor
  [graph id]
  (let [retval (get-node graph id)]
    (when-not-error (= :tensor (:type retval))
      "Node is not tensor"
      {:node-id id
       :node retval})
    retval))


(defn make-tensor
  [graph tens-name shape bufname & {:keys [sparse?]
                                    :or {sparse? false}}]
  (ensure-valid-tensor-shape graph shape)
  (let [buffer (get-buffer graph bufname)
        dimensions {:shape shape}]
    (add-node graph (->CompileTensor :tensor tens-name dimensions bufname (:dtype buffer) sparse?))))


(defn make-tensor-and-buffer
  [graph tens-name shape & {:keys [sparse? byte-offset? dtype]
                            :or {sparse? false byte-offset? false dtype :float32}}]
  (let [bufname (keyword (str (name tens-name) "-buffer"))]
    (-> graph
        (make-buffer bufname :dtype dtype :byte-offset? byte-offset?)
        (make-tensor tens-name shape bufname :sparse? sparse?))))


(defn check-sequence-max
  [num-seq ^long dim-len]
  (when-not-error (every? #(< (long %) dim-len))
    "Item in sequence out of range"
    {:num-seq num-seq
     :dimension-length dim-len})
  num-seq)

(defn apply-sequence-to-shape-item
  [select-arg shape-item]
  (cond
    (keyword? shape-item) select-arg
    (number? shape-item) (check-sequence-max select-arg shape-item)
    (sequential? shape-item) (let [select-arg (check-sequence-max select-arg (count shape-item))]
                               (mapv (vec shape-item) select-arg))
    :else (throw (ex-info "Unrecognized shape value type"
                          {:shape-value shape-item}))))


(defn check-value-max
  ^long [^long value ^long dim-len]
  (when-not-error (< value (long dim-len))
    "Select arg out of range"
    {:value value
     :dimension-length dim-len})
  value)

(defn apply-number-to-shape-item
  [^long select-arg shape-item]
  (cond
    (keyword? shape-item) [select-arg]
    (number? shape-item) [(check-value-max select-arg (long shape-item))]
    (sequential? shape-item) [(nth shape-item (check-value-max select-arg (count shape-item)))]
    :else
    (throw (ex-info "Unrecognized shape value type"
                    {:shape-value shape-item}))))

(defn generate-derived-node!
  "Generate a new node id and use new-node-fn to get a new node.
apply it to the graph atom and return the new node id."
  [*graph old-node]
  (loop [graph @*graph]
    (let [new-id (generate-derived-node-id old-node graph)
          new-node (assoc old-node :id new-id)]
      (if (compare-and-set! *graph graph (assoc-in graph [:nodes new-id] new-node))
        new-node
        (recur @*graph)))))


(defn add-edge!
  [graph* edge-op args output-tensors extra-keys]
  (let [new-edge (merge {:type edge-op
                         :args (vec args)
                         :input (->> (filter is-tensor? args)
                                     (mapv :id))
                         :output (mapv :id output-tensors)}
                        extra-keys)]
    (swap! graph* update :edges conj new-edge)
    new-edge))


(defn generate-op-result!
  [*graph op-name src-tensor res-dtype res-shape]
  (let [new-buffer (generate-derived-node! *graph {:type :buffer
                                                   :dtype res-dtype
                                                   :initial-shape res-shape
                                                   :id op-name
                                                   :byte-offset? false})]
    (generate-derived-node! *graph (->CompileTensor :tensor op-name {:shape res-shape} (:id new-buffer) res-dtype false))))


(defrecord CompileStream [*graph]
  fnp/PFunctionalBackend
  (select [stream item args]
    (let [graph @*graph
          item-shape (get-in item [:dimensions :shape])
          item-buffer (get-buffer graph (:bufname item))
          new-item-shape (mapv (fn [shape-item select-arg]
                                 (cond
                                   (= :all select-arg) shape-item
                                   (sequential? select-arg) (apply-sequence-to-shape-item select-arg shape-item)
                                   (number? select-arg) (apply-number-to-shape-item (long select-arg) shape-item)
                                   :else (throw (ex-info "Failed to recognize select argument" {:select-arg select-arg}))))
                               item-shape args)
          ;;Any vectors of numbers that are monotonically increasing
          ;;and don't start at 0 imply a byte offset.
          new-byte-offset? (->> new-item-shape
                                (filter #(and (sequential? %)
                                              (ct-dims/monotonically-increasing? %)
                                              (not= 0 (long (first %)))))
                                seq)
          ;;if the last shape item is a sequence of numbers that are not monotonically increasing
          ;;then we imply sparsity
          new-sparse? (and (sequential? (last new-item-shape))
                           (not (ct-dims/monotonically-increasing? (last new-item-shape))))
          ;;Reduce monotonically increasing sequences to just numbers as this will be the result of the runtime select.
          ;;We can't do this earlier else we cannot decifer if the buffer has a byte offset or not.
          new-item-shape (mapv (fn [shape-item]
                                 (if (and (sequential? shape-item)
                                          (ct-dims/monotonically-increasing? shape-item))
                                   (count shape-item)
                                   shape-item))
                               new-item-shape)
          new-buffer? (not= new-byte-offset? (:byte-offset? item-buffer))
          new-buffer-id (if new-buffer?
                          (:id (generate-derived-node! *graph (assoc item-buffer
                                                                     :byte-offset? new-byte-offset?
                                                                     :source-buffer (:id item-buffer))))
                          (:id item-buffer))
          retval (generate-derived-node! *graph (-> (assoc item
                                                           :bufname new-buffer-id
                                                           :sparse? (or (:sparse? item) new-sparse?))
                                                    (assoc-in [:dimensions :shape] new-item-shape)))]
      (add-edge! *graph :select [item args] [retval] {})
      retval))

  (transpose [stream item reorder-vec]
    (let [item-shape (vec (get-in item [:dimensions :shape]))
          new-shape (mapv item-shape reorder-vec)
          retval (generate-derived-node! *graph (assoc-in item [:dimensions :shape] new-shape))]
      (add-edge! *graph :transpose [item reorder-vec] [retval] {})
      retval))

  (static-cast [stream item dtype dest-shape]
    (let [graph @*graph
          old-buffer (get-buffer graph (:bufname item))
          new-shape (if dest-shape
                      dest-shape
                      ;;Get-shape performs a simplifying reduction
                      (mp/get-shape item))
          retval (generate-op-result! *graph :cast item dtype new-shape)]
      (add-edge! *graph :static-cast [item dtype dest-shape] [retval] {:dtype dtype})
      retval))

  (binary-op [stream lhs rhs op dest-shape]
    (let [graph @*graph
          arg-tensors (filter is-tensor? [lhs rhs])
          primary-tensor (first arg-tensors)
          result-shape (or dest-shape (mp/get-shape primary-tensor))
          result-dtype (or (when primary-tensor (ct/get-datatype primary-tensor))
                           ct/*datatype*)
          retval (generate-op-result! *graph :binary-op primary-tensor result-dtype result-shape)]
      (add-edge! *graph :binary-op [lhs rhs op dest-shape] [retval] {:op op})
      retval)))



(defn input-fn->graph
  [input-fn graph & input-fn-args]
  (let [*graph (atom graph)
        compile-stream (->CompileStream *graph)]
    (ct/with-stream compile-stream
      (apply input-fn input-fn-args))
    @*graph))


(defmulti edge-metadata
  "Return a map of metadata that describes an edge"
  :type)


(defmethod edge-metadata :select [edge] {:location :host})
(defmethod edge-metadata :transpose [edge] {:location :host})
(defmethod edge-metadata :static-cast [edge] {:location :device :operation :element-wise})
(defmethod edge-metadata :binary-op [edge] {:location :device :operation :element-wise})


(defn graph->host-device-graphs
  "Given a graph, split it into host and device graphs"
  [graph]
  (let [location-detector (comp #{:host} :location edge-metadata)
        host-edges (->> (edges graph)
                        (filter location-detector))
        device-edges (->> (edges graph)
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
  (let [leaf (get-tensor operation-graph (first (leaves operation-graph)))
        n-dims (count (mp/get-shape leaf))
        roots (mapv #(get-tensor operation-graph %) (roots operation-graph))]
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
                               (keyword? shape-item) true
                               (number? shape-item) true
                               (sequential? shape-item) (ct-dims/monotonically-increasing? shape-item)))
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


(defn custom-read-operation
  "A custom read bindings when the read operation falls outside the scope
  of what tvm supports."
  [index-vars tvm-tensor shape-vars stride-vars compile-time-shape]
  (api/tget tvm-tensor
            [(->> (map (fn [index-var ;; int32
                            shape-var ;; int32
                            stride-var ;; int32
                            shape-entry ;;number, sequence, keyword
                            ]
                         (let [index-var (if (and (sequential? shape-entry)
                                                  ;;If the sequence *is* monotonically increasing then whatever select produced
                                                  ;;them will be valid.  The only time we have a compile time mapping
                                                  ;;is if the sequence is *not* monotonically increasing.
                                                  (not (ct-dims/monotonically-increasing? shape-entry)))
                                           ;;ugghhh.  Build a large if-else lookup table and hope the compiler can build
                                           ;;a decent lookup table.
                                           (->> (map-indexed vector shape-entry)
                                                (reduce (fn [table [idx number]]
                                                          (let [bool-stmt (api/eq index-var (api/const (long idx) :dtype "int32"))
                                                                true-stmt (api/const (long number) :dtype "int32")
                                                                false-stmt table]
                                                            (api/select bool-stmt true-stmt false-stmt)))
                                                        index-var))
                                           index-var)]
                           (api/mul stride-var
                                    (api/mod index-var shape-var))))
                       index-vars shape-vars stride-vars compile-time-shape)
                  (reduce api/add))]))


(defn read-var
  [var res-dtype variable-map index-vars]
  (if (number? var)
    (api/const (tens-utils/dtype-cast var res-dtype))
    (let [read-op (get-in variable-map [(:id var) :read-operation])]
      (read-op index-vars))))


(defmulti tvm-eltwise-binding
  "Perform the specific tvm eltwise operation.
Return an updated variable map with the :read-operation updated with a function
that takes the index vars and produces a value for the approprate variable (result
of the edge operation.

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
        op-dtype (-> (get-tensor graph (first (:output edge)))
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
  "Produce the necessary read operation and binding information for the roots of the graph."
  (fn [read-op & args]
    (:read-type read-op)))


(defmethod read-op->variable-map-entry :custom-read
  [read-op n-dims operation-graph]
  (let [node (:node read-op)
        node-buffer (get-buffer operation-graph (:bufname node))
        datatype (dtype/get-datatype node)
        dtype-name (name datatype)
        node-name (name (:id node))
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
        node-buffer (get-buffer operation-graph (:bufname node))
        node-name (name (:id node))
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
  (when-not-error (= 1 (count (leaves operation-graph)))
    "Operation graph cannot handle multiple outputs"
    {:leaves (leaves operation-graph)})
  ;;The leaf defines the iteration variables
  (let [leaf (get-tensor operation-graph (first (leaves operation-graph)))
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
        output-name (name output-id)
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
        output-node (get-tensor operation-graph output-id)
        output-node-buffer (get-buffer operation-graph (:bufname output-node))
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
       (map #(if (is-tensor? %)
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
                                         (not (ct-dims/monotonically-increasing? select-arg)))
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
        roots-set (set (roots fn-graph))
        leaves-set (set (leaves fn-graph))
        total-input-list (c-set/union roots-set
                                      leaves-set
                                      ;;Because select,transpose,reshap happen on the host, if there is a select operation
                                      ;;on a piece of data that data buffer has to 'pop' out of the device graphs.
                                      ;;Else the device graphs can communicate tensor buffers to each other.
                                      (set (filter (set (roots host-graph))
                                                   (mapcat leaves device-op-graphs))))
        device-op-graphs (->> device-op-graphs
                              (map (fn [[idx device-op-graph]]
                                     (let [compiled-graph (create-tvm-operation device-op-graph)
                                           operation (:operation compiled-graph)
                                           op-schedule (condp = (:type operation)
                                                         :injective (tvm-reg/schedule-injective driver
                                                                                               (:operation operation)))]
                                       [idx
                                        (assoc-in compiled-graph [:operation :operation]
                                                  (api/schedule->lowered-function op-schedule
                                                                                  (:declaration-bind-list operation)
                                                                                  api/default-build-config
                                                                                  :name (str "fn_" idx)
                                                                                  :bind-map (:bind-map operation)))]))))
        compiled-functions (map #(get-in % [1 :operation :operation]) device-op-graphs)
        module (tvm-reg/->module driver compiled-functions)
        call-functions  (->> device-op-graphs
                             (mapv (fn [[idx device-op-graph]]
                                     (assoc device-op-graph
                                            :operation
                                            {:fn (c/get-module-function module (str "fn_" idx))
                                             :callsite-bind-list-fn (get-in device-op-graph
                                                                            [:operation :callsite-bind-list-fn])}))))
        final-fn (fn [arg-map]
                   (when-not-error (every? #(contains? arg-map %) total-input-list)
                     "Argmap does not include required argument"
                     {:argmap-keys (keys arg-map)
                      :required-args total-input-list})
                   (let [dev-argmap (->> (:edges host-graph)
                                         (reduce (fn [arg-map edge]
                                                   (assoc arg-map (first (:output edge))
                                                          (host-perform-edge arg-map host-graph edge)))
                                                 arg-map))]
                     (->> call-functions
                          (map (fn [op-graph]
                                 (let [tvm-fn (get-in op-graph [:operation :fn])
                                       callsite-binders (get-in op-graph [:operation :callsite-bind-list-fn])]
                                   (apply c/call-function tvm-fn (callsite-binders dev-argmap)))))
                          dorun)
                     nil))
        id->arg-definition-fn (fn [id-list]
                                (mapv #(let [tens (get-tensor fn-graph %)]
                                         {:id %
                                          :datatype (dtype/get-datatype tens)
                                          :shape (mp/get-shape tens)})
                                      id-list))]
    {:inputs (id->arg-definition-fn roots-set)
     :outputs (id->arg-definition-fn leaves-set)
     :intermediates (id->arg-definition-fn (c-set/difference total-input-list roots-set leaves-set))
     :fn! final-fn}))
