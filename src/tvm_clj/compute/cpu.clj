(ns tvm-clj.compute.cpu
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.base :as tvm-comp-base]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.datatype.base :as dtype]
            [tvm-clj.compute.device-buffer :as dbuf]
            [tvm-clj.compute.shared :as tvm-shared]
            [tech.compute.cpu.driver :as cpu-driver]
            [clojure.core.async :as async]
            [think.resource.core :as resource]))


(declare cpu-driver)

(defrecord CPUStream [device stream]
  drv/PStream
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count]
    (cpu-driver/with-stream-dispatch stream
      (tvm-shared/copy-array->array host-buffer host-offset device-buffer device-offset elem-count nil)))
  (copy-device->host [stream device-buffer device-offset
                      host-buffer host-offset elem-count]
    (cpu-driver/with-stream-dispatch stream
      (tvm-shared/copy-array->array device-buffer device-offset host-buffer host-offset elem-count nil)))
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count]
    (cpu-driver/with-stream-dispatch stream
      (tvm-shared/copy-array->array dev-a dev-a-off dev-b dev-b-off elem-count nil)))
  (memset [stream device-buffer device-offset elem-val elem-count]

    )
  (create-event [stream]
    (drv/create-event (.stream stream)))
  (sync-event [stream event]
    (drv/sync-event (.stream stream))))


(defn is-main-thread-cpu-stream?
  [^CPUStream stream]
  (cpu-driver/is-main-thread-cpu-stream? (.stream stream)))


(defmacro with-stream-dispatch
  [stream & body]
  `(cpu-driver/with-stream-dispatch (.stream stream)
     ~@body))


(defrecord CPUDevice [error-atom]
  tvm-comp-base/PDeviceInfo
  (device-type [this] (tvm-base/cpu-device-type))
  (device-id [this] 0)

  drv/PDevice
  (memory-info-impl [device]
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream-impl [device]
    (->CPUStream device (cpu-driver/cpu-stream device error-atom)))
  (allocate-device-buffer-impl [device elem-count elem-type]
    (dbuf/make-device-buffer-of-type device elem-type elem-count))
  (allocate-rand-buffer-impl [device elem-count]
    (dbuf/make-device-buffer-of-type device :float32 elem-count))

  drv/PDriverProvider
  (get-driver [dev] (cpu-driver))

  drv/PDeviceProvider
  (get-device [dev] dev)

  resource/PResource
  (release-resource [dev]))





(def cpu-devices
  (memoize
   (fn []
     [(->CPUDevice (atom nil))])))

(defrecord CPUDriver []
  drv/PDriver
  (get-devices [driver]
    (cpu-devices))
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    (hbuf/make-buffer-of-type elem-type elem-count)))


(def cpu-driver
  (memoize
   (fn [] (->CPUDriver))))
