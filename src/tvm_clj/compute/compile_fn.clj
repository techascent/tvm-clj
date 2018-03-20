(ns tvm-clj.compute.compile-fn
  (:require [tvm-clj.core :as c]
            [tvm-clj.api :as api]
            [tvm-clj.base :as b]
            [tvm-clj.compute.base :as comp-b]
            [tvm-clj.compute.tensor.functional-protocols :as fnp]
            [tech.compute.driver :as drv]
            [tech.compute.tensor.dimensions :refer [when-not-error] :as ct-dims]
            [tech.datatype.base :as dtype]
            [clojure.core.matrix.protocols :as mp]
            [tech.compute.tensor :as ct]
            [clojure.set :as c-set]
            [tvm-clj.compute.tensor-math :as tm]
            [clojure.string :as c-str]))


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
  [graph* edge-op args output-tensors]
  (let [new-edge {:type edge-op
                  :args (vec args)
                  :input (->> (filter is-tensor? args)
                              (mapv :id))
                  :output (mapv :id output-tensors)}]
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
      (add-edge! *graph :select [item args] [retval])
      retval))

  (transpose [stream item reorder-vec]
    (let [item-shape (vec (get-in item [:dimensions :shape]))
          new-shape (mapv item-shape reorder-vec)
          retval (generate-derived-node! *graph (assoc-in item [:dimensions :shape] new-shape))]
      (add-edge! *graph :tranpose [item reorder-vec] [retval])
      retval))

  (static-cast [stream item dtype dest-shape]
    (let [graph @*graph
          old-buffer (get-buffer graph (:bufname item))
          new-shape (if dest-shape
                      dest-shape
                      ;;Get-shape performs a simplifying reduction
                      (mp/get-shape item))
          retval (generate-op-result! *graph :cast item dtype new-shape)]
      (add-edge! *graph :static-cast [item dtype dest-shape] [retval])
      retval))

  (binary-op [stream lhs rhs op dest-shape]
    (let [graph @*graph
          arg-tensors (filter is-tensor? [lhs rhs])
          primary-tensor (first arg-tensors)
          result-shape (or dest-shape (mp/get-shape primary-tensor))
          result-dtype (or (when primary-tensor (ct/get-datatype primary-tensor))
                           ct/*datatype*)
          retval (generate-op-result! *graph :binary-op primary-tensor result-dtype result-shape)]
      (add-edge! *graph :binary-op [lhs rhs op dest-shape] [retval])
      retval)))



(defn compile-fn
  [input-fn graph & input-fn-args]
  (let [*graph (atom graph)
        compile-stream (->CompileStream *graph)]
    (ct/with-stream compile-stream
      (apply input-fn input-fn-args))
    @*graph))
