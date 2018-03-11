(ns tvm-clj.compute.tensor-math
  (:require [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [clojure.string :as s]
            [tvm-clj.api :as api]
            [tvm-clj.core :as core]
            [tvm-clj.base :as base]
            [tvm-clj.compute.base :as tvm-comp-base]
            [tvm-clj.compute.device-buffer :as dbuf]
            [tech.compute.tensor.utils :as tens-utils]
            [think.resource.core :as resource]
            [tech.compute.driver :as drv]
            [tvm-clj.compute.cpu]
            [tvm-clj.compute.gpu]
            [tech.compute.tensor.math :as tm])
  (:import [tvm_clj.compute.cpu CPUStream]
           [tvm_clj.compute.gpu GPUStream]))


(defonce ^:dynamic *fn-map* (atom {}))

(defrecord PRemoveFunction [device-type device-id fn-name]
  resource/PResource
  (release-resource [_] (swap! *fn-map* dissoc [device-type device-id fn-name])))


(defn y-dim-tvm-fn
  [n-dims detailed-fn]
  ;;The compute op turns the function argiments into iteration variables
  (with-meta (fn [& index-args]
               (detailed-fn index-args))
    {:arglists (->> (range n-dims)
                    (mapv #(symbol (str "i" %))))}))



(defn tensor-result-arg->mangle-str
  [tensor]
  (str (name (ct/get-datatype tensor))
       "_"
       (count (ct/shape tensor))
       "_"
       (if (ct/dense? tensor)
         "dense"
         "sparse")
       "_"
       (if (dbuf/has-byte-offset? tensor)
         "offset"
         "nooffset")))


(defn tensor-read-arg->mangle-str
  "For arguments where we ourselves are interpreting the argument."
  [tensor]
  (str (name (ct/get-datatype tensor))
       "_"
       (if (dbuf/has-byte-offset? tensor)
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
       (map (fn [[tensor variable]]
              (let [shape (ct/shape tensor)]
                [variable (api/declare-buffer
                           (:shape variable)
                           :dtype (name (ct/get-datatype tensor))
                           :name (or (:name variable) "unnamed")
                           :strides (if (ct/dense? tensor)
                                      nil
                                      (mapv #(api/variable (str "_stride_" %) :dtype "int32")
                                            (clojure.core/range (count shape))))
                           :elem-offset (if (dbuf/has-byte-offset? tensor)
                                          (api/variable "_elem_offset")
                                          nil))])))
       (into {})))


(defn get-or-create-fn
  [stream fn-name fn-create-fn]
  (let [device-type (tvm-comp-base/device-type stream)
        device-id (tvm-comp-base/device-id stream)]
    (if-let [retval (get @*fn-map* [device-type device-id fn-name])]
      retval
      (let [lowered-fn (fn-create-fn)
            module (tvm-comp-base/->module (drv/get-driver stream) [lowered-fn])
            retval (core/get-module-function module fn-name)]
        (swap! *fn-map* assoc fn-name retval)
        (resource/track (->PRemoveFunction device-type device-id fn-name))
        retval))))


(defn compute->lowered-function
  [stream fn-name compute-op arg-list tensor-arg-map]
  (let [schedule (api/create-schedule compute-op)]
    (when (tvm-comp-base/gpu-scheduling? (drv/get-driver stream))
      (let [compute-stage (get-in schedule [:stage_map compute-op])
            [bx tx] (api/split-stage-by-factor compute-stage (get-in compute-op [:axis 0]) 64)]
        (api/stage-bind compute-stage bx (api/name->thread-axis-iterator "blockIdx.x"))
        (api/stage-bind compute-stage tx (api/name->thread-axis-iterator "threadIdx.x"))))
    (api/schedule->lowered-function schedule arg-list
                                    api/default-build-config
                                    :name fn-name
                                    :bind-map (build-bind-map tensor-arg-map))))

(def device-datatype-map
  "https://github.com/dmlc/tvm/issues/984"
  {:uint8 :uint32
   :int8 :int32
   :uint16 :uint32
   :int16 :int32
   :uint64 :int64})


(defn- get-scalar-datatype
  [device fn-datatype]
  (if (tvm-comp-base/device-datatypes? device)
    (get device-datatype-map fn-datatype fn-datatype)
    fn-datatype))


(defn n-dim-compute-op
  [n-dims compute-fn]
  ;;Result shape has n-dims
  (api/compute (->> (range n-dims)
                    (mapv (fn [idx]
                            (api/variable (str "i" idx)))))
               (y-dim-tvm-fn n-dims compute-fn)))


(defn tensor-read-placeholder
  [tensor]
  (api/placeholder [(api/variable "_tens_ecount")] :dtype (name (ct/get-datatype tensor))))


(defn tensor-read-dims->vars
  [n-dims tensor arg-name]
  (when-not (<= (count (ct/shape tensor)) n-dims)
    (throw (ex-info "Read tensor can only equal or scatter into write tensor"
                    {:write-tensor-n-dims n-dims
                     :read-tensor-n-dims (count (ct/shape tensor))})))

  {:placeholder (tensor-read-placeholder tensor)
   :shape-stride-tuples (->> (range n-dims)
                                    (mapv (fn [idx]
                                            [(api/variable (str arg-name "_shape_" idx) :dtype "int32")
                                             (api/variable (str arg-name "_stride_" idx) :dtype "int32")])))})


(defn tensor-read
  "Perform the math required to get the absolute element offset from the shape/stride combined with the max shape variables"
  [placeholder index-vars shape-stride-tuples]
  (when-not (= (count index-vars)
               (count shape-stride-tuples))
    (throw (ex-info "Count of index vars must count of tensor shape"
                    {:index-var-count (count index-vars)
                     :shape-count (count shape-stride-tuples)})))
  ;;Generic broadcasting reduction for the destination indexes into any argument's shape.
  (api/tget placeholder
            [(->> (map (fn [index-var [shape stride]]
                         (api/mul stride
                                  (api/mod index-var shape)))
                       index-vars shape-stride-tuples)
                  (reduce api/add))]))


(defn explode-read-tensor
  [tensor max-shape]
  (let [tens-shape (ct-dims/left-pad-ones (ct/shape tensor) max-shape)
        tens-stride (ct-dims/extend-strides tens-shape (ct/strides tensor))]
    ;;Read tensors pass in their backing store so that we have generic broadcasting rules to effect.
    (concat [(ct/tensor->buffer tensor)]
            (map int tens-shape)
            (map int tens-stride))))




(defn assign-constant!
  [stream tensor value]
  (let [tensor (if (ct/dense? tensor)
                 (ct/as-vector tensor)
                 tensor)
        datatype (ct/get-datatype tensor)
        fn-name (get-fn-name "assign_constant" tensor)
        scalar-datatype (get-scalar-datatype (drv/get-driver stream) datatype)
        assign-fn (get-or-create-fn
                   stream fn-name
                   #(let [const-var (api/variable "const_val" :dtype (name scalar-datatype))
                          compute-op (n-dim-compute-op (count (ct/shape tensor))
                                                       (fn [& args]
                                                         (if (= scalar-datatype datatype)
                                                           const-var
                                                           (api/static-cast (name datatype) const-var))))
                          result (first (api/output-tensors compute-op))]
                      (compute->lowered-function stream fn-name compute-op
                                                 [result const-var] {tensor result})))]
    (tvm-comp-base/call-function stream assign-fn tensor (tens-utils/dtype-cast value scalar-datatype))))


(defn assign!
  "Broadcasting, marshalling assignment of rhs to lhs
lhs = rhs"
  [stream lhs rhs]
  (let [lhs-dtype (ct/get-datatype lhs)
        rhs-dtype (ct/get-datatype rhs)
        fn-name (get-fn-name "assign" lhs rhs)
        dest-datatype (ct/get-datatype lhs)
        n-dims (count (ct/shape lhs))
        max-shape (ct/shape lhs)
        assign-fn
        (get-or-create-fn
         stream fn-name
         #(let [dst-shape-vars (->> (range n-dims)
                                    (mapv (fn [idx]
                                            (api/variable (str "dest_shape_" idx) :dtype "int32"))))
                ;;Ignoring the fact the the shape at any index *could* be an array of data instead of
                ;;an integer...
                {rhs-placeholder :placeholder
                 rhs-shape-stride-tuples :shape-stride-tuples} (tensor-read-dims->vars n-dims rhs "rhs")
                compute-op (n-dim-compute-op
                            (count (ct/shape lhs))
                            (fn [index-vars]
                              (api/static-cast (name dest-datatype)
                                               (tensor-read rhs-placeholder
                                                            index-vars
                                                            rhs-shape-stride-tuples))))
                result (first (api/output-tensors compute-op))]
            (compute->lowered-function stream fn-name compute-op
                                       (->> (concat [result]
                                                    dst-shape-vars
                                                    [rhs-placeholder]
                                                    (map first rhs-shape-stride-tuples)
                                                    (map second rhs-shape-stride-tuples))
                                            vec)
                                       {lhs result})))]
    (apply tvm-comp-base/call-function
           stream assign-fn lhs (concat (map int (ct/shape lhs))
                                        (explode-read-tensor rhs max-shape)))))


(extend-protocol tm/TensorMath
  CPUStream
  (assign-constant! [stream tensor value]
    (assign-constant! stream tensor value))
  (assign! [stream lhs rhs]
    (assign! stream lhs rhs))

  GPUStream
  (assign-constant! [stream tensor value]
    (assign-constant! stream tensor value))
  (assign! [stream lhs rhs]
    (assign! stream lhs rhs)))
