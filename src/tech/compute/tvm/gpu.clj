(ns tech.compute.tvm.gpu
  (:require [tvm-clj.tvm-jna :as bindings]
            [tvm-clj.bindings.protocols :as tvm-proto]
            [tvm-clj.api :as api]
            [tvm-clj.jna.stream :as tvm-jna-stream]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.registry :as tvm-reg]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.datatype.base :as dtype]
            [tech.compute.tvm.device-buffer :as dbuf]
            [tech.compute.tvm :as tvm]
            [tech.resource :as resource]
            [tech.datatype.jna :as dtype-jna])
  (:import [com.sun.jna Pointer]))


(declare cuda-driver)
(declare opencl-driver)
(declare rocm-driver)
(declare make-gpu-device)


(defrecord GPUStream [device stream]
  tvm-proto/PToTVM
  (->tvm [_] stream)

  tvm-proto/PTVMDeviceType
  (device-type [_] (bindings/device-type device))

  tvm-proto/PTVMDeviceId
  (device-id [_] (bindings/device-id device))

  drv/PStream
  (copy-host->device [_ host-buffer host-offset
                      device-buffer device-offset elem-count]
    (dbuf/copy-device->device host-buffer host-offset
                              device-buffer device-offset
                              elem-count stream))
  (copy-device->host [_ device-buffer device-offset
                      host-buffer host-offset elem-count]
    (dbuf/copy-device->device device-buffer device-offset
                              host-buffer host-offset
                              elem-count stream))
  (copy-device->device [_ dev-a dev-a-off dev-b dev-b-off elem-count]
    (dbuf/copy-device->device dev-a dev-a-off
                              dev-b dev-b-off
                              elem-count stream))
  (sync-with-host [_]
    (bindings/sync-stream-with-host stream))
  (sync-with-stream [src-stream dst-stream]
    (when-not (= (tvm/device-type src-stream) (tvm/device-type dst-stream))
      (throw (ex-info "Cannot synchronize streams of two different device types"
                      {:src-device-type (tvm/device-type src-stream)
                       :dst-device-type (tvm/device-type dst-stream)})))
    (bindings/sync-stream-with-stream src-stream dst-stream))
  tvm-driver/PTVMStream
  (call-function [_ fn arg-list]
    (let [stream-ptr (-> (bindings/->tvm stream)
                         (dtype-jna/->ptr-backing-store))]
      (when-not (= 0 (Pointer/nativeValue stream-ptr))
        (bindings/set-current-thread-stream stream)))
    (apply fn arg-list))

  drv/PDeviceProvider
  (get-device [_] device)

  drv/PDriverProvider
  (get-driver [_] (drv/get-driver device))

  resource/PResource
  (release-resource [_] ))


(defrecord GPUDevice [driver ^long device-id supports-create?
                      default-stream resource-context]
  tvm-proto/PTVMDeviceType
  (device-type [this] (bindings/device-type driver))

  tvm-proto/PTVMDeviceId
  (device-id [this] device-id)

  drv/PDevice
  (memory-info [device]
    ;;This would be useful information
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream [device]
    (->GPUStream device (bindings/create-stream (bindings/device-type driver)
                                                device-id)))
  (allocate-device-buffer [device elem-count elem-type options]
    (dbuf/make-device-buffer-of-type device elem-type elem-count))
  (supports-create-stream? [device] supports-create?)
  (default-stream [device] @default-stream)
  (device->device-copy-compatible? [src dest]
    (let [src-device-type (bindings/device-type src)
          dst-device-type (when (satisfies? tvm-proto/PTVMDeviceType dest)
                            (bindings/device-type dest))]
      (or (= src-device-type dst-device-type)
          (= :cpu dst-device-type))))
  (acceptable-device-buffer? [device item]
    (tvm-driver/acceptable-tvm-device-buffer? item))

  drv/PDriverProvider
  (get-driver [dev] driver)

  drv/PDeviceProvider
  (get-device [dev] dev)

  resource/PResource
  (release-resource [_]
    (resource/release-resource resource-context)))


(defn- make-gpu-device
  "Never call this external; devices are centrally created and registered."
  [driver dev-id]
  (let [dev-type (tvm/device-type driver)
        {default-stream :return-value
         resource-seq :resource-seq}
        (resource/return-resource-seq
         (try
           (bindings/create-stream dev-type dev-id)
           (catch Throwable e
             (tvm-jna-stream/->StreamHandle dev-type dev-id (Pointer. 0)))))
        supports-create? (boolean default-stream)
        device (->GPUDevice driver dev-id supports-create?
                            (atom nil) (resource/->Releaser #(resource/release-resource-seq
                                                              resource-seq)))]
    (swap! (:default-stream device) (constantly (->GPUStream device default-stream)))
    device))


(def ^:private enumerate-devices
  (memoize
   (fn [driver]
     (->> (tvm/enumerate-device-ids (bindings/device-type driver))
          (mapv #(make-gpu-device driver %))))))


(def device-datatype-map
  "https://github.com/dmlc/tvm/issues/984"
  {:uint8 :uint32
   :int8 :int32
   :uint16 :uint32
   :int16 :int32
   :uint64 :int64})


(defrecord GPUDriver [device-type]
  tvm-proto/PTVMDeviceType
  (device-type [this] device-type)

  drv/PDriverProvider
  (get-driver [this] this)

  drv/PDriver
  (driver-name [this]
    (tvm-reg/tvm-driver-name (bindings/device-type this)))
  (get-devices [driver]
    (enumerate-devices driver))
  (allocate-host-buffer [driver elem-count elem-type options]
    (tvm/make-cpu-device-buffer elem-type elem-count))

  tvm-driver/PTVMDriver
  (device-id->device [driver dev-id]
    (if-let [retval (nth (drv/get-devices driver) dev-id)]
      retval
      (throw (ex-info "Device does not exist"
                      {:device-type device-type
                       :device-id dev-id}))))
  (gpu-scheduling? [_] true)
  (scalar-datatype->device-datatype [driver scalar-datatype]
    (get device-datatype-map scalar-datatype scalar-datatype))

  (schedule-injective! [driver stage compute-op {:keys [thread-count]}]
    ;;TODO - query device api for details like max thread count
    (let [device-max-thread-count 16]
      (api/stage-gpu-injective stage compute-op
                               :thread-count (or thread-count
                                                 device-max-thread-count))))
  (->module [driver sched-data-seq options]
    (api/schedules->fns sched-data-seq
                        :build-config (:build-config options)
                        :target-host (:target-host options)
                        :target-name device-type)))


(def gpu-device-types #{:cuda :opencl :rocm})


(def driver
  (memoize
   (fn [device-type]
     (when-not (gpu-device-types device-type)
       (throw (ex-info "Device type does not appear to be a gpu device"
                       {:device-type device-type})))
     (->GPUDriver device-type))))


(defn cuda-driver [] (driver :cuda))
(defn opencl-driver [] (driver :opencl))
(defn rocm-driver [] (driver :rocm))


(doseq [dev-type gpu-device-types]
  (tvm-reg/register-driver (driver dev-type)))
