(ns tech.compute.tvm.base)

(defprotocol PDeviceInfo
  (device-type [item])
  (device-id [item]))

(defprotocol PConvertToTVM
  (->tvm [item]))
