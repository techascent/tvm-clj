(ns tvm-clj.devices
  (:require [tvm-clj.impl.definitions :as definitions]
            [tvm-clj.impl.fns.runtime :as runtime-fns]))


(defn device-id-exists?
  [device-type device-id]
  (runtime-fns/GetDeviceAttr
   (definitions/device-type->device-type-int device-type)
   (long device-id) (definitions/device-attribute-map :exists)))
