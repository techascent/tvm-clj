(ns tech.compute.tvm.shared
  (:require [tvm-clj.core :as tvm-core]))


(defprotocol PDeviceInfo
  (device-type [item])
  (device-id [item]))
