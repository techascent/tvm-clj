(ns tech.compute.tvm.gpu
  (:require [tvm-clj.tvm-bindings :as bindings]
            [tvm-clj.api :as api]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.registry :as tvm-reg]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.datatype.base :as dtype]
            [tech.compute.tvm.device-buffer :as dbuf]
            [tech.compute.tvm :as tvm]
            [tech.resource :as resource])
  (:import [tvm_clj.tvm runtime$TVMStreamHandle]))


(declare cuda-driver)
(declare opencl-driver)
(declare rocm-driver)
(declare make-gpu-device)


(defrecord GPUStream [device stream]
  bindings/PToTVM
  (->tvm [_] stream)

  tvm-driver/PTVMDeviceType
  (device-type [_] (tvm/device-type device))

  tvm-driver/PTVMDeviceId
  (device-id [_] (tvm/device-id device))

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
    (bindings/sync-stream-with-host (-> (tvm/device-type device)
                                        (bindings/device-type->device-type-int))
                                    (tvm/device-id device)
                                    stream))
  (sync-with-stream [src-stream dst-stream]
    (when-not (= (tvm/device-type src-stream) (tvm/device-type dst-stream))
      (throw (ex-info "Cannot synchronize streams of two different device types"
                      {:src-device-type (tvm/device-type src-stream)
                       :dst-device-type (tvm/device-type dst-stream)})))
    (bindings/sync-stream-with-stream (-> (tvm/device-type device)
                                          (bindings/device-type->device-type-int))
                                      (tvm/device-id device)
                                      (bindings/->tvm src-stream)
                                      (bindings/->tvm dst-stream)))
  tvm-driver/PTVMStream
  (call-function [_ fn arg-list]
    (when (.address (bindings/->tvm stream))
      (bindings/set-current-thread-stream (bindings/device-type->device-type-int
                                           (tvm/device-type device))
                                          (tvm/device-id device)
                                          stream))
    (apply fn arg-list))

  drv/PDeviceProvider
  (get-device [_] device)

  drv/PDriverProvider
  (get-driver [_] (drv/get-driver device))

  resource/PResource
  (release-resource [_] ))


(defrecord GPUDevice [driver ^long device-id supports-create?
                      default-stream resource-context]
  tvm-driver/PTVMDeviceType
  (device-type [this] (tvm-driver/device-type driver))

  tvm-driver/PTVMDeviceId
  (device-id [this] device-id)

  drv/PDevice
  (memory-info [device]
    ;;This would be useful information
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream [device]
    (->GPUStream device (bindings/create-stream (tvm-driver/device-type driver)
                                                device-id)))
  (allocate-device-buffer [device elem-count elem-type options]
    (dbuf/make-device-buffer-of-type device elem-type elem-count))
  (supports-create-stream? [device] supports-create?)
  (default-stream [device] @default-stream)
  (device->device-copy-compatible? [src dest]
    (let [src-device-type (tvm-driver/device-type src)
          dst-device-type (when (satisfies? tvm-driver/PTVMDeviceType dest)
                            (tvm-driver/device-type dest))]
      (or (= src-device-type dst-device-type)
          (= :cpu dst-device-type))))

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
  (let [dev-type (-> (tvm/device-type driver)
                     (bindings/device-type->device-type-int))
        {default-stream :return-value
         resource-seq :resource-seq}
        (resource/return-resource-seq
         (try
           (bindings/create-stream dev-type dev-id)
           (catch Throwable e
             (bindings/->StreamHandle dev-type dev-id (runtime$TVMStreamHandle.)))))
        supports-create? (boolean default-stream)
        device (->GPUDevice driver dev-id supports-create? (atom nil)
                            (resource/make-resource #(resource/release-resource-seq
                                                      resource-seq)))]
    (swap! (:default-stream device) (constantly (->GPUStream device default-stream)))
    device))


(def ^:private enumerate-devices
  (memoize
   (fn [driver]
     (->> (tvm/enumerate-device-ids (tvm-driver/device-type driver))
          (mapv #(make-gpu-device driver %))))))


(def device-datatype-map
  "https://github.com/dmlc/tvm/issues/984"
  {:uint8 :uint32
   :int8 :int32
   :uint16 :uint32
   :int16 :int32
   :uint64 :int64})


(defrecord GPUDriver [^long device-type]
  tvm-driver/PTVMDeviceType
  (device-type [this] (bindings/device-type-int->device-type
                       device-type))

  drv/PDriverProvider
  (get-driver [this] this)

  drv/PDriver
  (driver-name [this]
    (tvm-reg/tvm-driver-name (tvm-driver/device-type this)))
  (get-devices [driver]
    (enumerate-devices driver))
  (allocate-host-buffer [driver elem-count elem-type options]
    (tvm/make-cpu-device-buffer elem-type elem-count))

  tvm-driver/PTVMDriver
  (device-id->device [driver dev-id]
    (if-let [retval (nth (drv/get-devices driver) dev-id)]
      retval
      (throw (ex-info "Device does not exist"
                      {:device-type (bindings/device-type-int->device-type device-type)
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
                        :target-name (bindings/device-type-int->device-type
                                      device-type))))


(def gpu-device-types #{:cuda :opencl :rocm})


(def driver
  (memoize
   (fn [device-type]
     (let [device-type (long (if (number? device-type)
                               device-type
                               (bindings/device-type->device-type-int device-type)))]
       (when-not (gpu-device-types (bindings/device-type-int->device-type device-type))
         (throw (ex-info "Device type does not appear to be a gpu device"
                         {:device-type (bindings/device-type-int->device-type
                                        device-type)})))
       (->GPUDriver device-type)))))


(defn cuda-driver [] (driver :cuda))
(defn opencl-driver [] (driver :opencl))
(defn rocm-driver [] (driver :rocm))


(doseq [dev-type gpu-device-types]
  (tvm-reg/register-driver (driver dev-type)))
