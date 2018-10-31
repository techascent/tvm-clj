(ns tech.compute.tvm.cpu
  (:require [tvm-clj.tvm-jna :as bindings]
            [tvm-clj.api :as api]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.registry :as tvm-reg]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.compute.tvm.device-buffer :as dbuf]
            [tech.compute.cpu.driver :as cpu-driver]
            [tech.compute.registry :as registry]
            [tech.resource :as resource]
            [tech.datatype :as dtype]
            [tech.compute.tvm :as tvm]
            [tech.compute.registry :as registry]
            [tech.compute :as compute]
            [tech.datatype.jna :as dtype-jna])
  ;;Setup so these objects (and anything derived from them)
  ;;auto-link to the tvm cpu compute device.
  (:import [org.bytedeco.javacpp.Pointer]
           [com.sun.jna Pointer]
           [tech.datatype.jna TypedPointer]))


(declare driver)

(defrecord CPUStream [device-fn stream]
  drv/PStream
  (copy-host->device [_ host-buffer host-offset
                      device-buffer device-offset elem-count]
    (cpu-driver/with-stream-dispatch stream
      (dbuf/copy-device->device host-buffer host-offset
                                device-buffer device-offset elem-count nil)))
  (copy-device->host [_ device-buffer device-offset
                      host-buffer host-offset elem-count]
    (cpu-driver/with-stream-dispatch stream
      (dbuf/copy-device->device device-buffer device-offset
                                host-buffer host-offset elem-count nil)))
  (copy-device->device [_ dev-a dev-a-off dev-b dev-b-off elem-count]
    (cpu-driver/with-stream-dispatch stream
      (dbuf/copy-device->device dev-a dev-a-off
                                dev-b dev-b-off elem-count nil)))
  (sync-with-host [_]
    (drv/sync-with-host stream))
  (sync-with-stream [_ dst-stream]
    (drv/sync-with-stream stream (:stream dst-stream)))

  resource/PResource
  (release-resource [_] )

  tvm-driver/PTVMStream
  (call-function [_ fn arg-list]
    (cpu-driver/with-stream-dispatch stream
      (apply fn arg-list)))

  drv/PDriverProvider
  (get-driver [_] (drv/get-driver (device-fn)))

  drv/PDeviceProvider
  (get-device [_] (device-fn))

  bindings/PTVMDeviceType
  (device-type [this] (bindings/device-type (device-fn)))

  bindings/PTVMDeviceId
  (device-id [this] (bindings/device-id (device-fn)))

  cpu-driver/PToCPUStream
  (->cpu-stream [this] stream))


(declare make-cpu-device-buffer)

(defrecord CPUDevice [error-atom default-stream]
  bindings/PTVMDeviceId
  (device-id [this] 0)

  bindings/PTVMDeviceType
  (device-type [_] :cpu)

  drv/PDevice
  (memory-info [device]
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream [device]
    (->CPUStream device (cpu-driver/cpu-stream device error-atom)))
  (allocate-device-buffer [device elem-count elem-type options]
    (make-cpu-device-buffer elem-type elem-count))
  (supports-create-stream? [device] true)
  (default-stream [device] @default-stream)
  (device->device-copy-compatible? [src dest]
    ;;Is it a tvm device?
    (boolean (satisfies? bindings/PTVMDeviceType dest)))

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
           default-stream (->CPUStream (constantly device)
                                       (cpu-driver/main-thread-cpu-stream))]
       (swap! (:default-stream device) (constantly default-stream))
       [device]))))

(defrecord CPUDriver []
  drv/PDriverProvider
  (get-driver [this] this)
  drv/PDriver
  (driver-name [driver]
    (tvm-reg/tvm-driver-name :cpu))
  (get-devices [driver]
    (cpu-devices))
  (allocate-host-buffer [driver elem-count elem-type options]
    (make-cpu-device-buffer elem-type elem-count))

  tvm-driver/PTVMDriver
  (device-id->device [driver dev-id]
    (when-not (= 0 dev-id)
      (throw (ex-info "CPU device types only have device id 0" {})))
    (first (drv/get-devices driver)))
  (gpu-scheduling? [driver] false)
  (scalar-datatype->device-datatype [driver scalar-datatype] scalar-datatype)
  (schedule-injective! [driver stage compute-op options]
    (apply api/stage-cpu-injective stage compute-op options))
  (->module [driver sched-data-seq options]
    (api/schedules->fns sched-data-seq
                        :build-config (:build-config options)
                        :target-host (:target-host options)
                        :target-name :llvm))

  bindings/PTVMDeviceType
  (device-type [_] :cpu))

(def driver
  (memoize
   (fn [] (->CPUDriver))))

(defn ptr->device-buffer
  [ptr & {:keys [dtype]}]
  (let [dtype (or dtype (dtype/get-datatype ptr))
        shape [(dtype/ecount ptr)]
        device (first (cpu-devices))]
    (bindings/pointer->tvm-ary ptr :cpu 0 dtype shape nil 0)))


(defn make-cpu-device-buffer
  [elem-type elem-count]
  (dbuf/make-device-buffer-of-type
   (compute/default-device (driver))
   elem-type elem-count))


(defmacro extend-tvm-bindings
  [item-cls]
  `(clojure.core/extend
       ~item-cls
     bindings/PToTVM
     {:->tvm (fn [item#]
               (bindings/pointer->tvm-ary
                item#
                :cpu
                0
                (dtype/get-datatype item#)
                (dtype/shape item#)
                nil
                0))}
     bindings/PJVMTypeToTVMValue
     {:->tvm-value (fn [item#]
                     (-> item#
                         bindings/->tvm
                         bindings/->tvm-value))}
     bindings/PTVMDeviceType
     {:device-type (fn [item#] :cpu)}
     bindings/PTVMDeviceId
     {:device-id (fn [item#] 0)}))


(extend-tvm-bindings com.sun.jna.Pointer)
(extend-tvm-bindings org.bytedeco.javacpp.Pointer)
(extend-tvm-bindings TypedPointer)


(tvm-reg/register-driver (driver))

;;Set the compute cpu driver name
(registry/set-cpu-driver-name! (-> (driver)
                                   drv/driver-name))
