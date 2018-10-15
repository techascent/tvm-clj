(ns tvm-clj.compute.shared
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :refer [->tvm] :as tvm-base]
            [tech.resource :as resource]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype])
  (:import [tvm_clj.tvm runtime runtime$TVMStreamHandle]))


(defn enumerate-devices
  [^long device-type]
  (->> (range)
       (take-while #(= 1 (tvm-core/device-exists? device-type %)))))


(defn maybe-sub-buffer
  [buf offset elem-count]
  (if-not (and (= 0 (long offset))
               (= (dtype/ecount buf) (long elem-count)))
    (drv/sub-buffer buf offset elem-count)
    buf))


(defn copy-device->device
  [dev-a dev-a-off dev-b dev-b-off elem-count stream]
  (resource/with-resource-context
    (let [dev-a (maybe-sub-buffer dev-a dev-a-off elem-count)
          dev-b (maybe-sub-buffer dev-b dev-b-off elem-count)]
      (tvm-core/copy-array-to-array! (->tvm dev-a) (->tvm dev-b) stream))))
