(ns tvm-clj.compute.base
  (:require [tvm-clj.core :as tvm-core]))

(defn cpu-device-type
  ^long []
  (tvm-core/device-type->device-type-int :cpu))

(defn cuda-device-type
  ^long []
  (tvm-core/device-type->device-type-int :cuda))

(defn opencl-device-type
  ^long []
  (tvm-core/device-type->device-type-int :opencl))

(defn rocm-device-type
  ^long []
  (tvm-core/device-type->device-type-int :rocm))

(defprotocol PDeviceInfo
  (device-type [item])
  (device-id [item]))

(defprotocol PDeviceIdToDevice
  (device-id->device [driver device-id]))

;;Mapping from integer device types to drivers implementing that type.
(defonce ^:dynamic *device-types->drivers* (atom {}))

(defn add-device-type
  [^long device-type driver]
  (swap! *device-types->drivers* assoc device-type driver))

(defn get-device
  [device-type ^long device-id]
  (let [device-type (long (if (number? device-type)
                               device-type
                               (tvm-core/device-type->device-type-int device-type)))])
  (if-let [driver (get @*device-types->drivers* device-type)]
    (device-id->device driver device-id)
    (throw (ex-info "Failed to find driver for device type:"
                    {:device-type device-type}))))
