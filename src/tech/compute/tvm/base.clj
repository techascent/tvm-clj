(ns tech.compute.tvm.base
  (:require [tvm-clj.core :as tvm-core]))

(defn cpu-device-type
  ^long []
  (tvm-core/device-type->device-type-int :cpu))

(defprotocol PDeviceInfo
  (device-type [item])
  (device-id [item]))

(defprotocol PDeviceIdToDevice
  (device-id->device [driver device-id]))

(defprotocol PConvertToTVM
  (->tvm [item]))

;;Mapping from integer device types to drivers implementing that type.
(def ^:dynamic *device-types->drivers* (atom {}))

(defn add-device-type
  [^long device-type driver]
  (swap! *device-types->drivers* assoc device-type driver))

(defn get-device
  [^long device-type ^long device-id]
  (if-let [driver (get @*device-types->drivers* device-type)]
    (device-id->device driver device-id)
    (throw (ex-info "Failed to find driver for device type:"
                    {:device-type device-type}))))
