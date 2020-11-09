(ns tvm-clj.devices
  (:require [tvm-clj.impl.definitions :as definitions]
            [tvm-clj.impl.protocols :as tvm-proto]
            [tvm-clj.impl.fns.runtime :as runtime-fns]
            [tvm-clj.impl.dl-tensor :as dl-tensor]
            [tvm-clj.impl.stream :as stream]))


(defn device-id-exists?
  [device-type device-id]
  (= 1 (runtime-fns/GetDeviceAttr
        (definitions/device-type->device-type-int device-type)
        (long device-id) (definitions/device-attribute-map :exists))))



(defn device-tensor
  "Allocate a device tensor."
  ([shape datatype device-type device-id options]
   (dl-tensor/allocate-device-array shape datatype device-type
                                    device-id options))
  ([shape datatype device-type device-id]
   (device-tensor shape datatype device-type device-id nil)))



(defn copy-tensor!
  "Copy a src tensor to a destination tensor."
  [src-tens dest-tens stream]
  (dl-tensor/copy-array-to-array! src-tens dest-tens stream)
  dest-tens)



(defn stream
  "Create a device stream of execution."
  [device-type device-id]
  (stream/create-stream device-type device-id))


(defn set-current-thread-stream
  [stream]
  (stream/set-current-thread-stream stream))


(defn sync-stream-with-host
  "Synchonize the device stream with the host"
  [stream]
  (stream/sync-stream-with-host stream))
