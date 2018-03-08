(ns tvm-clj.compute.cpu-tensor-math
  (:require [tvm-clj.compute.cpu :as cpu]
            [tvm-clj.compute.base :as tvm-comp-base]
            [tech.compute.tensor.math :as tm]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.compute.driver :as drv]
            [tvm-clj.api :as api]
            [tvm-clj.core :as core]
            [clojure.string :as s]
            [think.resource.core :as resource]
            [tech.compute.tensor.utils :as tens-utils])
  (:import [tvm_clj.compute.cpu CPUStream]))

(defn assign-constant-fn
  [dst-dtype const-value]
  (api/const const-value :dtype (name dst-dtype)))


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
         "d"
         "s")))


(defn get-fn-name
  [stem & tensor-args]
  (s/join "-" (concat [stem]
                      (map tensor-arg->mangle-str
                           tensor-args))))


(defonce ^:dynamic *fn-map* (atom {}))

(defrecord PRemoveFunction [fn-name]
  resource/PResource
  (release-resource [_] (swap! *fn-map* dissoc fn-name)))


(defn get-or-create-fn
  [stream fn-name fn-create-fn]
  (if-let [retval (get @*fn-map* fn-name)]
    retval
    (let [lowered-fn (fn-create-fn)
          module (tvm-comp-base/->module (drv/get-driver stream) [lowered-fn])
          retval (core/get-module-function module fn-name)]
      (swap! *fn-map* assoc fn-name retval)
      (resource/track (->PRemoveFunction fn-name))
      retval)))



(extend-protocol tm/TensorMath
  CPUStream
  (assign-constant! [stream tensor value]
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
                            result (first (api/output-tensors compute-op))
                            schedule (api/create-schedule compute-op)]
                        (api/schedule->lowered-function schedule [result const-var]
                         api/default-build-config :name fn-name)))]
      (tvm-comp-base/call-function stream assign-fn tensor (tens-utils/dtype-cast value datatype)))))
