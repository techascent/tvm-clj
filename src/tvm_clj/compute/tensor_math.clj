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
  (let [fn-args (->> (range n-dims)
                     (mapv #(symbol (str "i" %))))]
    (with-meta (fn [& args]
                 (detailed-fn n-dims fn-args))
      {:arglists fn-args})))



(defn tensor-arg->mangle-str
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


(defn get-fn-name
  [stem & tensor-args]
  (s/join "_" (concat [stem]
                      (map tensor-arg->mangle-str
                           tensor-args))))

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


(defn assign-constant!
  [stream tensor value]
  (let [tensor (if (ct/dense? tensor)
                 (ct/as-vector tensor)
                 tensor)
        datatype (ct/get-datatype tensor)
        n-dims (count (ct/shape tensor))
        fn-name (get-fn-name "assign_constant" tensor)
        assign-fn (get-or-create-fn
                   stream fn-name
                   #(let [index-vars (->> (range n-dims)
                                          (mapv (fn [idx]
                                                  (api/variable (str "i" idx)))))
                          const-var (api/variable "const_val" :dtype (name datatype))
                          compute-op (api/compute index-vars
                                                  (y-dim-tvm-fn
                                                   n-dims
                                                   (fn [& args]
                                                     const-var)))
                          result (first (api/output-tensors compute-op))]
                      (compute->lowered-function stream fn-name compute-op
                                                 [result const-var] {tensor result})))]
    (tvm-comp-base/call-function stream assign-fn tensor (tens-utils/dtype-cast value datatype))))


(extend-protocol tm/TensorMath
  CPUStream
  (assign-constant! [stream tensor value]
    (assign-constant! stream tensor value))
  GPUStream
  (assign-constant! [stream tensor value]
    (assign-constant! stream tensor value)))
