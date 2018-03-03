(ns tech.compute.tvm.cpu
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.shared :as shared]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.datatype.base :as dtype]
            [tech.compute.tvm.host-buffer :as hbuf]
            [tech.compute.tvm.device-buffer :as dbuf]
            [tech.compute.tvm.shared :as tvm-shared]))


(defn cpu-device-type
  ^long []
  (tvm-core/device-type->device-type-int :cpu))

(declare cpu-driver)

(defrecord CPUEvent [])

(defrecord CPUStream [device])

(defrecord CPUDevice []
  tvm-shared/PDeviceInfo
  (device-type [this] (cpu-device-type))
  (device-id [this] 0)
  drv/PDevice
  (memory-info-impl [device]
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream-impl [device]
    (->CPUStream device))
  (allocate-device-buffer-impl [device elem-count elem-type]
    ()
    "Allocate a device buffer.  This is the generic unit of data storage used for computation.")
  (allocate-rand-buffer-impl [device elem-count]
    "Allocate a buffer used for rands.  Random number generation in general needs a divisible-by-2 element count
and a floating point buffer (cuda cuRand limitation)"))

(def cpu-devices
  (memoize
   (fn []
     [(->CPUDevice)])))

(defrecord CPUDriver []
  drv/PDriver
  (get-devices [driver]
    (cpu-devices))
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    (hbuf/make-buffer-of-type elem-type elem-count)))


(def cpu-driver
  (memoize
   (fn [] (->CPUDriver))))
