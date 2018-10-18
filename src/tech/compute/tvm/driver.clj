(ns tech.compute.tvm.driver
  "Additional protocols for tvm drivers, devices, and streams.
Centralized registring of drivers allowing a symbolic name->driver table."
  (:require [tvm-clj.tvm-bindings :as bindings]
            [tvm-clj.api :as tvm-api]))

(defn cpu-device-type
  ^long []
  (bindings/device-type->device-type-int :cpu))

(defn cuda-device-type
  ^long []
  (bindings/device-type->device-type-int :cuda))

(defn opencl-device-type
  ^long []
  (bindings/device-type->device-type-int :opencl))

(defn rocm-device-type
  ^long []
  (bindings/device-type->device-type-int :rocm))


(defprotocol PTVMDeviceId
  (device-id [item]
    "Return the tvm integer device of a given device, buffer or stream."))


(defprotocol PTVMDeviceType
  (device-type [item]))


(defprotocol PTVMDriver
  (device-id->device [driver device-id])
  (gpu-scheduling? [driver])
  ;;https://github.com/dmlc/tvm/issues/984
  (device-datatypes? [driver])
  ;;Basic injective scheduling.  Updates stage
  (schedule-injective! [driver stage compute-op options])
  ;;Build the module.  See api/schedules->fns
  (->module [driver sched-data-seq options]))


(defprotocol PTVMStream
  (call-function [stream fn arg-list]))
