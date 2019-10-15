(ns tech.libs.tvm.driver
  "Additional protocols for tvm drivers, devices, and streams.
Centralized registring of drivers allowing a symbolic name->driver table."
  (:require [tvm-clj.tvm-jna :as bindings]
            [tvm-clj.bindings.protocols :as tvm-proto]
            [tech.v2.datatype.protocols :as dtype-proto]))


(defprotocol PTVMDriver
  (device-id->device [driver device-id])
  (gpu-scheduling? [driver])
  ;;https://github.com/dmlc/tvm/issues/984
  (scalar-datatype->device-datatype [driver scalar-datatype])
  ;;Basic injective scheduling.  Updates stage
  (schedule-injective! [driver stage compute-op options])
  ;;Build the module.  See api/schedules->fns
  (->module [driver sched-data-seq options]))


(defn has-byte-offset? [buffer]
  (not= 0 (bindings/byte-offset buffer)))


(defprotocol PTVMStream
  (call-function [stream fn arg-list]))


(defn acceptable-tvm-device-buffer?
  [item]
  (and (dtype-proto/convertible-to-buffer-desc? item)
       (every? #(satisfies? % item)
               [tvm-proto/PToTVM
                tvm-proto/PJVMTypeToTVMValue
                tvm-proto/PByteOffset
                tvm-proto/PTVMDeviceType
                tvm-proto/PTVMDeviceId])))


(defn acceptable-tvm-host-buffer?
  "Things that pass this will work in copy-host->device."
  [item]
  (and (acceptable-tvm-device-buffer? item)
       (= :cpu (tvm-proto/device-type item))))
