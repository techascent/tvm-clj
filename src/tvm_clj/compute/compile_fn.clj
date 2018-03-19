(ns tvm-clj.compute.compile-fn
  (:require [tvm-clj.core :as c]
            [tvm-clj.api :as api]
            [tvm-clj.base :as b]
            [tvm-clj.compute.base :as comp-b]
            [tvm-clj.compute.tensor.functional-protocols :as fnp]
            [tech.compute.driver :as drv]
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.datatype.base :as dtype]
            [clojure.core.matrix.protocols :as mp]
            [tech.compute.tensor :as ct]
            [clojure.set :as c-set]
            [tvm-clj.compute.tensor-math :as tm]))


(defrecord BindTensor [dimensions strides? byte-offset? datatype argname]
  dtype/PDatatype
  (get-datatype [this] datatype)

  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (:shape dimensions))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape}))))))


(defn bind-tensor
  [n-dims argname & {:keys [strides? byte-offset? datatype]
                     :or {strides? true byte-offset? false datatype ct/*datatype*
                          }}]
  (->BindTensor {:shape (vec (repeat n-dims :all))
                 :index-order (ct-dims/reversev (range n-dims))}
                strides? byte-offset? datatype {:name argname
                                               :idx 0}))


(defn valid-select-arg?
  "Args can either be the keyword all, an integer, or a sequence"
  [arg-val]
  (cond
    (= :all arg-val) true
    (sequential? arg-val) true
    (number? arg-val) true
    :else false))


(def ^:dynamic *preamble* (atom []))
;;Map of compile-time tensors to runtime tensors during runtime
(def ^:dynamic *runtime-variables* (atom {}))
;;Map of compile-time tensors to compile-time information
(def ^:dynamic *compile-bindings* (atom {}))
(def ^:dynamic *function-call* (atom []))
(def ^:dynamic *id* (atom 0))

(defn append-preamble-fn!
  [fn]
  (swap! *preamble* conj fn))

(defn next-id!
  ^long []
  (swap! *id* inc))

(defn unique-argname!
  [argname]
  (assoc argname :id (next-id!)))

(defn increment-argname
  [argname]
  (assoc argname :idx (next-id!)))

(defn argname->str
  [argname]
  (str (:name argname) "_" (:idx argname)))

(defn store-fn-result!
  [{arg-key :argname} arg-val]
  (swap! *runtime-variables* assoc-in [arg-key :tensor] arg-val))

(defn find-fn-arg
  [{arg-key :argname}]
  (if-let [arg-val (get-in @*runtime-variables* [arg-key :tensor])]
    arg-val
    (throw (ex-info "Failed to find argument for key:" {:arg-key arg-key}))))


(defn get-or-create-shape-strides-buffer
  "Generate a flat-bound buffer and shape strides and
push everything into the function call.  This is used if we want to use a custom striding
algorithm such as broadcasting or when there is an index vector (constant or dynamic).
Returns map of
{:placeholder tensor
 :shape-stride-tuples [[shape-var stride-var][shape-var stride-var]]
}"
  [{:keys [argname] :as tensor} n-max-shape-dims]
  (if-let [bindings (get-in @*compile-bindings* [argname :flat-bindings])]
    bindings
    (let [compile-shape (mp/get-shape tensor)
          n-dims (long n-max-shape-dims)
          {:keys [placeholder shape-stride-tuples]} (tm/tensor-read-dims->vars tensor)
          retval {:placeholder placeholder
                  :shape-stride-tuples shape-stride-tuples}]
      ;;Add these arguments to the function call
      (swap! *function-call* conj {:compile-list (concat [placeholder]
                                                         (map first shape-stride-tuples)
                                                         (map second shape-stride-tuples))
                                   :bind-map #(vector placeholder (api/declare-buffer
                                                                   (:shape placeholder)
                                                                   :dtype (name (dtype/get-datatype tensor))
                                                                   :name (argname->str argname)
                                                                   :elem-offset (if (:byte-offset? tensor)
                                                                                  (api/variable "_elem_offset")
                                                                                  nil)))
                                   :runtime-list #(tm/explode-read-tensor (find-fn-arg tensor) n-dims)})
      ;;Make sure we do not re-generate these arguments
      (swap! *compile-bindings* assoc-in [argname :flat-bindings] retval)
      retval)))


(defn valid-tvm-shape?
  [shape]
  (->> shape
       ;;TVM does not support arbitrary lists of indexes as a shape.
       (remove #(or (= :all %)
                    (number? %)))
       nil?))


(defn extend-runtime-tensor-shape
  [tensor n-dims]
  (let [tens-shape (tm/left-pad-ones (get-in tensor [:dimensions :shape]) n-dims)]
    (assoc tensor
           :dimensions {:shape tens-shape
                        :strides (ct-dims/extend-strides tens-shape
                                                         (get-in tensor [:dimensions :strides]))})))


(defn get-or-create-tensor-buffer
  "Generate a bound buffer using the bind-tensor's settings for byte-offset and stride but
  tvm's shape and stride system.  Implies no broadcasting nor index vector."
  [{:keys [argname] :as tensor} n-max-shape-dims]
  (if-let [bindings (get-in @*compile-bindings* [argname :full-bindings])]
    bindings
    (let [compile-shape (mp/get-shape tensor)]
      (when-not (valid-tvm-shape? compile-shape)
        (throw (ex-info "Shape is not a valid tvm shape"
                        {:shape compile-shape})))
      (let [n-dims (long n-max-shape-dims)
            shape-vars (->> (range n-dims)
                            (mapv (fn [idx]
                                    (api/variable (str (argname->str argname) "_shape_" idx :dtype "int32")))))
            placeholder (api/placeholder shape-vars :dtype (name (dtype/get-datatype tensor)) :name (argname->str argname))]
        (swap! *function-call* conj {:compile-list [placeholder]
                                     :bind-map #(vector placeholder (api/declare-buffer
                                                                     (:shape placeholder)
                                                                     :dtype (name (dtype/get-datatype tensor))
                                                                     :name (argname->str argname)
                                                                     :strides (if (:sparse? tensor)
                                                                                (mapv (fn [idx]
                                                                                        (api/variable (str "_stride_" idx)
                                                                                                      :dtype "int32"))
                                                                                      (clojure.core/range n-max-shape-dims))
                                                                                nil)
                                                                     :elem-offset (if (:byte-offset? tensor)
                                                                                    (api/variable "_elem_offset")
                                                                                    nil)))
                                     :runtime-list #(-> (find-fn-arg tensor)
                                                        (extend-runtime-tensor-shape n-dims)
                                                        (assoc :sparse? (:sparse? tensor)))})
        (swap! *compile-bindings* assoc-in [argname :full-bindings] placeholder)
        placeholder))))


(defn preamble-operation
  [item result preamble-fn]
  (append-preamble-fn! #(store-fn-result! result (preamble-fn (find-fn-arg item))))
  result)


(defn runtime-process-select-args
  "For each arg return the number if it should be a number,
monotonically-increasing definition if it is monotonically-increasing vector
or just the vector if it is a sequence."
  [shape args]
  (ct-dims/when-not-error (= (count shape)
                             (count args))
    "Shape arg count mismatch"
    {:shape-count (count shape)
     :arg-count (count args)})
  (map (fn [dim arg]
         (cond
           (= arg :all)
           {:type :monotonically-increasing
            :start 0
            :count dim}
           (number? arg) arg
           (sequential? arg)
           (if (ct-dims/monotonically-increasing? arg)
             (ct-dims/sequence->monotonic-definition arg)
             (vec arg))
           :else
           (throw (ex-info "Invalid select arg type."))))
       shape args))


(defn- runtime-select
  "Sort of like select but we only really want to calculate the byte-offset of the item.
Runtime item is a tvm tensor.  Args are the arguments to select."
  [runtime-item args]
  (ct/apply-select-result runtime-item
                          (ct-dims/select-detail (ct/tensor->dimensions runtime-item) args
                                                 :arg-processor runtime-process-select-args)))



(defn cleanup-index-order
  "In order for in-place arbitrary transpose to work at compile time we need to maintain an index order
  for indexing into the object.  If we elide a dimension (due to it being hardcoded to '1')
  then our index order will have gaps in it and thus will be invalid.  A valid index order is one
  that is one that is a vector of the integers [0-ndims) and is exactly n-dims in length.
  This is a vector->vector translation."
  [index-order]
  (let [index-set (set index-order)
        [reindex-map index-gap]
        (->> (range (+ 1 (apply max index-order)))
             (reduce (fn [[reindex-map index-gap] next-index]
                       (if (index-set next-index)
                         [(assoc reindex-map next-index (- next-index index-gap)) index-gap]
                         [reindex-map (inc index-gap)]))
                     [{} 0]))]
    (mapv reindex-map index-order)))


(defn select-arg-imply-byte-offset?
  [args]
  (->> args
       (filter (fn [arg]
                 (cond
                   (= arg :all) false
                   (number? arg) (not= arg 0)
                   (sequential? arg) (and (ct-dims/monotonically-increasing? arg)
                                          (not= 0 (first arg)))
                   :else
                   (throw (ex-info "Unsupported argument type for select" {:argument arg})))))
       seq
       boolean))


(defn valid-dest-shape?
  "Shapes used for destination shapes must be full specified; meaning all integer constants.
They cannot be :all or [1 2 3 5]"
  [dest-shape]
  (every? number? dest-shape))


(defn ensure-valid-dest-shape
  [dest-shape]
  (when-not (or (nil? dest-shape) (valid-dest-shape? dest-shape))
    (throw (ex-info "Destination shape is invalid"
                    {:dest-shape dest-shape})))
  dest-shape)


(defn arg-shape->result-shape
  "Convert any sequential lists to integers"
  [arg-shape]
  (mapv (fn [shape-item]
          (if (vector? shape-item)
            (count shape-item)
            shape-item))
        arg-shape))


(defn ensure-tvm-shape
  [dimensions]
  (update dimensions :tvm-shape
          (fn [tvm-shape]
            (if tvm-shape
              tvm-shape
              (mapv (fn [shape-item]
                      (cond
                        (number? shape-item)
                        (api/const (int shape-item))
                        (= :all shape-item)
                        (api/variable "shape_var" :dtype "int32")
                        :else
                        (throw (ex-info "Failed to map shape data"
                                        {:shape-arg shape-item})))))))))


(defrecord CompileStream [^long device-type]
  fnp/PFunctionalBackend
  (select [stream item args]
    (let [item-shape (mp/get-shape item)]
      (ct-dims/when-not-error (= (count (seq (filter valid-select-arg? args)))
                                 (count args))
        "Invalid select arguments found:"
        {:invalid-arguments (remove valid-select-arg? args)})
      (ct-dims/when-not-error (= (count args)
                                 (count item-shape))
        "Select dimensionality does not match item dimensionality"
        {:select-arg-count (count args)
         :item-shape-count (count item-shape)})
      (let [shape-idx-order-pairs (->> (map (fn [old-arg new-arg idx-order]
                                              (if (= new-arg :all)
                                                [old-arg idx-order]
                                                ;;This should be a more intelligent combining
                                                (cond
                                                  (= old-arg :all) [new-arg idx-order]
                                                  :else (throw (ex-info "Unimplemented" {})))))
                                            (get-in item [:dimensions :shape])
                                            args
                                            (get-in item [:dimensions :index-order]))
                                       (remove (comp number? first)))]

        (preamble-operation
         item
         (increment-argname
          (assoc item
                 :dimensions {:shape (mapv first shape-idx-order-pairs)
                              :index-order (->> (map second shape-idx-order-pairs)
                                                cleanup-index-order)}
                 ;;Being safe for now.  It is detectable whether a byte offset is required and whether the result
                 ;;is dense considering the input and the exact shape translation but that is an error prone
                 ;;algorithm not necessary at this time.
                 :byte-offset? (boolean (or (:byte-offset item)
                                            (select-arg-imply-byte-offset? args)))
                 :strides? true))
         #(runtime-select % args)))))

  (transpose [stream item reorder-vec]
    (let [item-shape (mp/get-shape item)
          item-indexes (get-in item [:dimensions :index-order])]
      (ct-dims/when-not-error (= (count item-shape)
                                 (count reorder-vec))
        "Reorder vector must match dimension length"
        {:shape-count (count item-shape)
         :reorder-vec-count (count reorder-vec)})
      (preamble-operation
       item
       (increment-argname
        (assoc item
               :dimensions {:shape (mapv #(get item-shape %) reorder-vec)
                            :index-order (mapv #(get item-indexes %) reorder-vec)}
               :strides? true))
       #(ct/transpose % reorder-vec))))

  (static-cast [stream item dtype dest-shape]
    (let [result-dimensions (ensure-tvm-shape
                             (if (ensure-valid-dest-shape dest-shape)
                               {:shape dest-shape
                                :tvm-shape (mapv #(api/const (int %)) dest-shape)}
                               {:shape (arg-shape->result-shape (mp/get-shape item))
                                :tvm-shape (get-in item [:dimensions :tvm-shape])}))
          compute-op (api/compute (:tvm-shape result-dimensions)
                                  (tm/y-dim-tvm-fn
                                   (count (:shape result-dimensions))
                                   (fn [index-vars]
                                     (api/static-cast
                                      (name dtype)
                                      (tensor-read item index-vars
                                                   item-shape-stride-tuples (mp/get-shape item))))))
          result (first (api/output-tensors compute-op))]
      (assoc (->BindTensor result-dimensions false false dtype (increment-argname (:argname item)))
             :tensor result)))

  (binary-op [stream lhs rhs op dest-shape]))
