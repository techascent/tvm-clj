(ns tech.compute.tvm.cpu
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.shared :as shared]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.datatype.base :as dtype]))


(defn cpu-device-type
  ^long []
  (tvm-core/device-type->device-type-int :cpu))


(defrecord CPUEvent [])

(defrecord CPUStream [device])

(defrecord CPUDevice [])

(def cpu-devices
  (memoize
   (fn []
     [(->CPUDevice)])))

(defrecord CPUDriver []
  drv/PDriver
  (get-devices [driver]
    (cpu-devices))
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    (jcpp-dtype/make-pointer-of-type elem-count (shared/tvm-type->dtype-type elem-type))))
