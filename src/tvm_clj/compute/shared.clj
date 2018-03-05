(ns tvm-clj.compute.shared
  (:require [tvm-clj.core :as tvm-core]
            [think.resource.core :as resource]
            [tech.compute.driver :as drv]
            [tvm-cljcompute.base :refer [->tvm] :as tvm-comp-base]
            [tech.datatype.base :as dtype])
  (:import [tvm_clj.tvm runtime runtime$TVMStreamHandle]))


(defn enumerate-devices
  [^long device-type]
  (->> (range)
       (take-while #(tvm-core/device-exists? device-type %))))


(defn copy-device->device
  [dev-a dev-a-off dev-b dev-b-off elem-count tvm-stream]
  (let [dev-a (drv/sub-buffer dev-a dev-a-off elem-count)
        dev-b (drv/sub-buffer dev-b dev-b-off elem-count)
        tvm-stream (if tvm-stream tvm-stream (runtime$TVMStreamHandle.))]
    (tvm-core/copy-array-to-array! (->tvm dev-a)  (->tvm dev-b) :stream-hdl tvm-stream)))


(defn copy-host->device
  [host-a host-a-off dev-b dev-b-off elem-count]
  (let [host-a (drv/sub-buffer host-a host-a-off elem-count)
        dev-b (drv/sub-buffer dev-b dev-b-off elem-count)]
    (tvm-core/copy-to-array! (->tvm host-a) (->tvm dev-b)
                             (* elem-count (dtype/datatype->byte-size
                                            (dtype/get-datatype host-a))))))


(defn copy-device->host
  [dev-a dev-a-off host-b host-b-off elem-count]
  (let [host-b (drv/sub-buffer host-b host-b-off elem-count)
        dev-a (drv/sub-buffer dev-a dev-a-off elem-count)]
    (tvm-core/copy-from-array! (->tvm dev-a) (->tvm host-b)
                               (* elem-count (dtype/datatype->byte-size
                                              (dtype/get-datatype host-b))))))
