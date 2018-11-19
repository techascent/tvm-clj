(ns tech.compute.tvm.tensor-math
  (:require [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.compute.tensor.utils :refer [when-not-error]]
            [clojure.string :as s]
            [tvm-clj.api :as api]
            [tvm-clj.tvm-jna :as bindings]
            [tech.resource :as resource]
            [tech.compute.tvm.cpu :as cpu]
            [tech.compute.tvm.gpu]
            [tech.compute.tensor.math :as tm]
            [tech.datatype :as dtype]
            [clojure.core.matrix :as m]
            [tech.compute.tvm :as tvm]
            [tech.compute :as compute]
            [tech.datatype.javacpp :as jcpp-dtype]
            [tech.compute.cpu.driver :as cpu-driver]
            [tech.datatype.jna :as dtype-jna]
            ;;Need this for the fallbacks
            [tech.compute.cpu.tensor-math :as cpu-tm])
  (:import [tech.compute.tvm.cpu CPUStream]
           [tech.compute.tvm.gpu GPUStream]
           [java.util UUID]))


(defonce ^:dynamic *fn-map* (atom {}))


(defn y-dim-tvm-fn
  [n-dims detailed-fn]
  ;;The compute op turns the function argiments into iteration variables
  (with-meta (fn [& index-args]
               (detailed-fn index-args))
    {:arglists (->> (range n-dims)
                    (mapv #(symbol (str "i" %))))}))



(defn tensor-result-arg->mangle-str
  [tensor]
  (let [shape (ct/shape tensor)]
    (when-let [invalid-items (->> shape
                                  (filter (or sequential? ct/tensor?))
                                  seq)]
      (throw (ex-info "Destination shape has invalid shape entries"
                      {:invalid-entries invalid-items})))
    (str (name (ct/get-datatype tensor))
         "_"
         (count (ct/shape tensor))
         "_"
         (if (ct/dense? tensor)
           "dense"
           "sparse")
         "_"
         (if (tvm/has-byte-offset? tensor)
           "offset"
           "nooffset"))))


(defn tensor-read-arg->mangle-str
  "For arguments where we ourselves are interpreting the argument."
  [tensor]
  (str (name (ct/get-datatype tensor))
       (if (tvm/has-byte-offset? tensor)
         "offset"
         "nooffset")))


(defn get-fn-name
  [stem dest & args]
  (s/join "_" (concat [stem (tensor-result-arg->mangle-str dest)]
                      (map tensor-read-arg->mangle-str args))))


(defn build-bind-map
  "The special cases in the bind map must relate to the name mangling
  of the function"
  [tensor-var-map]
  (->> tensor-var-map
       (map-indexed (fn [idx [tensor variable]]

                      (let [[result? tensor variable] (if (= tensor :result)
                                                        [true (first variable)
                                                         (second variable)]
                                                        [false tensor variable])
                            shape (ct/shape tensor)]
                        [variable (api/declare-buffer
                                   (if result?
                                     (:shape variable)
                                     [(api/variable (str "unnamed__dimension_" idx))])
                                   :dtype (name (ct/get-datatype tensor))
                                   :name (or (:name variable) (str "unnamed" idx))
                                   :strides (if (or (not result?)
                                                    (ct/dense? tensor))
                                              nil
                                              (mapv #(api/variable (str "_stride_" %)
                                                                   :dtype "int32")
                                                    (clojure.core/range (count shape))))
                                   :elem-offset (if (tvm/has-byte-offset? tensor)
                                                  (api/variable (str "_elem_offset"))
                                                  nil))])))
       (into {})))


(defn get-or-create-fn
  [stream fn-name fn-create-fn]
  (let [device-type (tvm/device-type stream)
        device-id (tvm/device-id stream)]
    (if-let [retval (get @*fn-map* [device-type device-id fn-name])]
      retval
      (let [retval (fn-create-fn)]
        (swap! *fn-map* assoc fn-name retval)
        (resource/make-resource #(swap! *fn-map* dissoc fn-name))
        retval))))


(defn- get-scalar-datatype
  [device fn-datatype]
  (-> (compute/->driver device)
      (tvm/scalar-datatype->device-datatype fn-datatype)))


(defn n-dim-compute-op
  [n-dims compute-fn & {:keys [name]
                        :or {name "compute_op"}}]
  ;;Result shape has n-dims
  (api/compute (->> (range n-dims)
                    (mapv (fn [idx]
                            (api/variable (str name "_i" idx)))))
               (y-dim-tvm-fn n-dims compute-fn)
               name))


(defn n-dims->shape-stride-tuples
  [n-dims arg-name]
  (->> (range n-dims)
       (mapv (fn [idx]
               [(api/variable (str arg-name "_shape_" idx) :dtype "int32")
                (api/variable (str arg-name "_stride_" idx) :dtype "int32")]))))


(defn tensor-read-placeholder
  [tensor arg-name]
  (api/placeholder [(api/variable "_tens_ecount")] arg-name
                   :dtype (name (ct/get-datatype tensor))))


(defn tensor-read-dims->vars
  [n-dims tensor arg-name]
  (when-not (<= (count (ct/shape tensor)) n-dims)
    (throw (ex-info "Read tensor can only equal or scatter into write tensor"
                    {:write-tensor-n-dims n-dims
                     :read-tensor-n-dims (count (ct/shape tensor))})))

  {:placeholder (tensor-read-placeholder tensor arg-name)
   :shape-stride-tuples (n-dims->shape-stride-tuples n-dims arg-name)})


(defn tensor-read
  "Perform the math required to get the absolute element offset from the shape/stride
  combined with the max shape variables"
  [placeholder index-vars shape-stride-tuples]
  (when-not (= (count index-vars)
               (count shape-stride-tuples))
    (throw (ex-info "Count of index vars must count of tensor shape"
                    {:index-var-count (count index-vars)
                     :shape-count (count shape-stride-tuples)})))
  ;;Generic broadcasting reduction for the destination indexes into any argument's
  ;;shape.
  (api/tget placeholder
            [(->> (map (fn [index-var [shape stride]]
                         (api/mul stride
                                  (api/mod index-var shape)))
                       index-vars shape-stride-tuples)
                  (reduce api/add))]))


(defn left-pad-ones
  [shape-vec n-dims]
  (concat (repeat (- (long n-dims) (count shape-vec)) 1)
          shape-vec))


(defn explode-read-tensor
  [tensor n-dims]
  (let [tens-shape (left-pad-ones (ct/shape tensor) n-dims)
        tens-stride (ct-dims/extend-strides tens-shape (ct/strides tensor))]
    ;;Read tensors pass in their backing store so that we have generic broadcasting
    ;;rules to effect.
    (concat [(ct/tensor->buffer tensor)]
            (map int tens-shape)
            (map int tens-stride))))


(defn- compile-operation
  [driver fn-name compute-op arglist bind-map]
  (let [schedule (api/create-schedule compute-op)
        _ (tvm/schedule-injective! driver schedule compute-op {})
        mod-fns (tvm/->module driver [{:schedule schedule
                                           :name fn-name
                                           :arglist arglist
                                           :bind-map (build-bind-map bind-map)}])]
    (get-in mod-fns [:fn-map fn-name])))




(defn assign-constant!
  [stream tensor value]
  (let [tensor (if (ct/dense? tensor)
                 (ct/as-vector tensor)
                 tensor)
        datatype (ct/get-datatype tensor)
        fn-name (keyword (get-fn-name "assign_constant" tensor))
        scalar-datatype (get-scalar-datatype (compute/->driver stream) datatype)
        assign-fn (get-or-create-fn
                   stream fn-name
                   #(let [const-var (api/variable "const_val"
                                                  :dtype (name scalar-datatype))
                          compute-op (n-dim-compute-op (count (ct/shape tensor))
                                                       (fn [& args]
                                                         (if (= scalar-datatype datatype)
                                                           const-var
                                                           (api/static-cast
                                                            (name datatype)
                                                            const-var))))
                          result (first (api/output-tensors compute-op))]
                      (compile-operation (compute/->driver stream)
                                         fn-name compute-op
                                         [result const-var]
                                         {:result [tensor result]})))]
    (tvm/call-function stream assign-fn tensor (dtype/cast value scalar-datatype))))


(defn assign!
  "Broadcasting, marshalling assignment of rhs to lhs
lhs = rhs"
  [stream lhs rhs]
  (let [lhs-dtype (ct/get-datatype lhs)
        rhs-dtype (ct/get-datatype rhs)
        fn-name (keyword (get-fn-name "assign" lhs rhs))
        dest-datatype (ct/get-datatype lhs)
        n-dims (count (ct/shape lhs))
        max-shape (ct/shape lhs)
        assign-fn
        (get-or-create-fn
         stream fn-name
         #(let [;;Ignoring the fact the the shape at any index *could* be an array of
                ;;data instead of an integer...
                {rhs-placeholder :placeholder
                 rhs-shape-stride-tuples :shape-stride-tuples}
                (tensor-read-dims->vars n-dims rhs "rhs")
                compute-op (n-dim-compute-op
                            (count (ct/shape lhs))
                            (fn [index-vars]
                              (api/static-cast (name dest-datatype)
                                               (tensor-read rhs-placeholder
                                                            index-vars
                                                            rhs-shape-stride-tuples))))
                result (first (api/output-tensors compute-op))]
            (compile-operation (compute/->driver stream)
                               fn-name compute-op
                               (->> (concat [result]
                                            [rhs-placeholder]
                                            (map first rhs-shape-stride-tuples)
                                            (map second rhs-shape-stride-tuples))
                                    vec)
                               {:result [lhs result]
                                rhs rhs-placeholder})))]
    (apply tvm/call-function
           stream assign-fn lhs (explode-read-tensor rhs (count max-shape)))))


(defn- can-generate-code?
  "We can generate code if:
1.  The dest tensor's dimensions match the max-dimensions.
2.  The source tensors are all direct."
  [dest-tensor & args]
  (let [all-dims (concat [(ct/tensor->dimensions dest-tensor)]
                         (->> args
                              (filter ct/tensor?)
                              (map ct/tensor->dimensions)))
        {:keys [max-shape]} (apply ct-dims/dimension-seq->max-shape all-dims)
        result-shape (ct/shape dest-tensor)]
    ;;We can broadcast into a result but that is it.
    (and (= max-shape result-shape)
         ;;Iteration through each dimension is monotonically increasing
         (every? #(every? number? (:shape %)) all-dims))))


(defn- ensure-code-generation
  [dest-tensor & args]
  (when-not-error (apply can-generate-code? dest-tensor args)
    "Cannot generate code for this type of indexing yet"
    {}))


(defmacro cpu-stream-fallback
  [fn-name stream-item & args]
  `(let [cpu-stream# (:stream ~stream-item)]
     (if (can-generate-code? ~@args)
       (cpu-driver/with-stream-dispatch
         cpu-stream#
         (~fn-name ~stream-item ~@args))
       (~(symbol "tm" (str fn-name)) cpu-stream# ~@args))))


(defmacro cpu-fallback-impl
  [tm-fn stream & args]
  `(~tm-fn (:stream ~stream) ~@args))


(extend-protocol tm/TensorMath
  CPUStream
  (assign-constant! [stream tensor value]
    (cpu-stream-fallback assign-constant! stream tensor value))
  (assign! [stream lhs rhs]
    (cpu-stream-fallback assign! stream lhs rhs))
  (unary-accum! [stream dest alpha op]
    (cpu-fallback-impl tm/unary-accum! stream dest alpha op))
  (unary-op! [stream dest x alpha op]
    (cpu-fallback-impl tm/unary-op! stream dest x alpha op))
  (binary-accum-constant! [stream dest dst-alpha scalar operation reverse-operands?]
    (cpu-fallback-impl tm/binary-accum-constant! stream dest dst-alpha
                       scalar operation reverse-operands?))

  (binary-op-constant! [stream dest x x-alpha scalar operation reverse-operands?]
    (cpu-fallback-impl tm/binary-op-constant! stream dest x x-alpha
                       scalar operation reverse-operands?))

  (binary-accum! [stream dest dest-alpha y y-alpha operation
                  reverse-operands? dest-reduction?]
    (cpu-fallback-impl tm/binary-accum! stream dest dest-alpha
                       y y-alpha operation reverse-operands? dest-reduction?))

  (binary-op! [stream dest x x-alpha y y-alpha operation]
    (cpu-fallback-impl tm/binary-op! stream dest x x-alpha
                       y y-alpha operation))

  (ternary-op! [stream dest
                x x-alpha
                y y-alpha
                z z-alpha
                operation]
    (cpu-fallback-impl tm/ternary-op! stream dest
                       x x-alpha
                       y y-alpha
                       z z-alpha
                       operation))

  (ternary-op-constant! [stream dest a a-alpha b b-alpha
                         constant operation arg-order]
    (cpu-fallback-impl tm/ternary-op-constant! stream dest a a-alpha b b-alpha
                       constant operation arg-order))

  (ternary-op-constant-constant! [stream dest a a-alpha
                                  const-1 const-2 operation
                                  arg-order]
    (cpu-fallback-impl tm/ternary-op-constant-constant! stream dest a a-alpha
                       const-1 const-2 operation arg-order))

  (unary-reduce! [stream output input-alpha input op]
    (cpu-fallback-impl tm/unary-reduce! stream output input-alpha input op))

  (gemm! [stream
          c-buf c-colstride
          trans-a? trans-b? alpha
          a-buf a-row-count a-col-count a-colstride
          b-buf b-col-count b-colstride
          beta]
    (tvm/call-function stream #(bindings/global-function
                                "tvm.contrib.cblas.matmul"
                                a-buf b-buf c-buf trans-a? trans-b?
                                alpha beta)))

  (rand! [stream dest distribution]
    (cpu-fallback-impl tm/rand! stream dest distribution))

  GPUStream
  (gemm! [stream
          c-buf c-colstride
          trans-a? trans-b? alpha
          a-buf a-row-count a-col-count a-colstride
          b-buf b-col-count b-colstride
          beta]
    (let [device-type (tvm/device-type (compute/->device stream))]
      (cond
        (= :cuda device-type)
        (tvm/call-function stream
                           #(bindings/global-function
                             "tvm.contrib.cublas.matmul"
                             a-buf b-buf c-buf trans-a? trans-b?
                             alpha beta))
        :else
        (throw (ex-info "gemm not implemented for this device type"
                        {:device-type device-type})))))

  (assign-constant! [stream tensor value]
    (ensure-code-generation tensor)
    (assign-constant! stream tensor value))
  (assign! [stream lhs rhs]
    (ensure-code-generation lhs rhs)
    (assign! stream lhs rhs)))


(defn as-cpu-tensor
  [data & {:keys [shape datatype]}]
  (when (dtype-jna/typed-pointer? data)
    (let [shape (or shape (ct/shape data))
          datatype (or datatype (ct/get-datatype data))]
      (-> (cpu/ptr->device-buffer data :dtype datatype)
          (ct/ensure-tensor)))))
