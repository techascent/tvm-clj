(ns tvm-clj.compute.cpu
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tvm-clj.api :as api]
            [tech.compute.driver :as drv]
            [tvm-clj.compute.base :as tvm-comp-base]
            [tvm-clj.compute.device-buffer :as dbuf]
            [tvm-clj.compute.shared :as tvm-shared]
            [tech.compute.cpu.driver :as cpu-driver]
            [think.resource.core :as resource]))


(declare driver)

(defrecord CPUStream [device-fn stream]
  drv/PStream
  (copy-host->device [_ host-buffer host-offset
                      device-buffer device-offset elem-count]
    (cpu-driver/with-stream-dispatch stream
      (tvm-shared/copy-device->device host-buffer host-offset
                                      device-buffer device-offset elem-count nil)))
  (copy-device->host [_ device-buffer device-offset
                      host-buffer host-offset elem-count]
    (cpu-driver/with-stream-dispatch stream
      (tvm-shared/copy-device->device device-buffer device-offset
                                      host-buffer host-offset elem-count nil)))
  (copy-device->device [_ dev-a dev-a-off dev-b dev-b-off elem-count]
    (cpu-driver/with-stream-dispatch stream
      (tvm-shared/copy-device->device dev-a dev-a-off
                                      dev-b dev-b-off elem-count nil)))
  (sync-with-host [_]
    (drv/sync-with-host stream))
  (sync-with-stream [_ dst-stream]
    (drv/sync-with-stream stream (:stream dst-stream)))

  resource/PResource
  (release-resource [_] )

  tvm-comp-base/PTVMStream
  (call-function-impl [_ fn arg-list]
    (cpu-driver/with-stream-dispatch stream
      (apply tvm-core/call-function fn arg-list)))

  drv/PDriverProvider
  (get-driver [_] (drv/get-driver (device-fn)))

  drv/PDeviceProvider
  (get-device [_] (device-fn))

  tvm-comp-base/PDeviceInfo
  (device-type [this] (tvm-comp-base/device-type (device-fn)))
  (device-id [this] (tvm-comp-base/device-id (device-fn))))

(defn is-main-thread-cpu-stream?
  [^CPUStream stream]
  (cpu-driver/is-main-thread-cpu-stream? (.stream stream)))

(defmacro with-stream-dispatch
  [stream & body]
  `(cpu-driver/with-stream-dispatch (.stream stream)
     ~@body))

(defrecord CPUDevice [error-atom default-stream]
  tvm-comp-base/PDeviceInfo
  (device-type [this] (tvm-comp-base/cpu-device-type))
  (device-id [this] 0)

  drv/PDevice
  (memory-info-impl [device]
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream-impl [device]
    (->CPUStream device (cpu-driver/cpu-stream device error-atom)))
  (allocate-device-buffer-impl [device elem-count elem-type]
    (dbuf/make-cpu-device-buffer elem-type elem-count))
  (allocate-rand-buffer-impl [device elem-count]
    (dbuf/make-cpu-device-buffer :float32 elem-count))
  (supports-create-stream? [device] true)
  (default-stream [device] @default-stream)

  drv/PDriverProvider
  (get-driver [dev] (driver))

  drv/PDeviceProvider
  (get-device [dev] dev)

  resource/PResource
  (release-resource [dev]))


(def cpu-devices
  (memoize
   (fn []
     (let [device (->CPUDevice (atom nil) (atom nil))
           default-stream (->CPUStream (constantly device) (cpu-driver/main-thread-cpu-stream))]
       (swap! (:default-stream device) (constantly default-stream))
       [device]))))

(defrecord CPUDriver []
  drv/PDriver
  (get-devices [driver]
    (cpu-devices))
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    (dbuf/make-cpu-device-buffer elem-type elem-count))

  tvm-comp-base/PDeviceIdToDevice
  (device-id->device [driver dev-id]
    (when-not (= 0 dev-id)
      (throw (ex-info "CPU device types only have device id 0" {})))
    (first (drv/get-devices driver)))

  tvm-comp-base/PCompileModule
  (gpu-scheduling? [driver] false)
  (device-datatypes? [driver] false)
  (schedule-injective [driver compute-op]
    ;;For injective we fuse all dimensions and then run them all in parallel.
    (let [schedule (api/create-schedule [compute-op])
          stage (get-in schedule [:stage_map compute-op])
          op-axis (:axis compute-op)
          fused-axis (apply api/stage-fuse stage op-axis)]
      (api/stage-parallel stage fused-axis)
      schedule))
  (->module-impl [driver lowered-fn-seq build-config]
    (api/lowered-functions->module lowered-fn-seq build-config :target-name :llvm)))

(def driver
  (memoize
   (fn [] (->CPUDriver))))

(tvm-comp-base/add-device-type (tvm-core/device-type->device-type-int :cpu) (driver))
