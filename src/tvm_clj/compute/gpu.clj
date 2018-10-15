(ns tvm-clj.compute.gpu
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tvm-clj.api :as api]
            [tech.compute.driver :as drv]
            [tvm-clj.compute.registry :as tvm-reg]
            [tech.datatype.base :as dtype]
            [tvm-clj.compute.device-buffer :as dbuf]
            [tvm-clj.compute.shared :as tvm-shared]
            [tech.resource :as resource])
  (:import [tvm_clj.tvm runtime$TVMStreamHandle]))


(declare cuda-driver)
(declare opencl-driver)
(declare rocm-driver)
(declare make-gpu-device)


(defrecord GPUStream [device stream]
  tvm-base/PToTVM
  (->tvm [_] stream)

  tvm-reg/PDriverInfo
  (device-type [_] (tvm-reg/device-type device))

  tvm-reg/PDeviceInfo
  (device-id [_] (tvm-reg/device-id device))

  drv/PStream
  (copy-host->device [_ host-buffer host-offset
                      device-buffer device-offset elem-count]
    (tvm-shared/copy-device->device host-buffer host-offset
                                    device-buffer device-offset elem-count stream))
  (copy-device->host [_ device-buffer device-offset
                      host-buffer host-offset elem-count]
    (tvm-shared/copy-device->device device-buffer device-offset
                                    host-buffer host-offset elem-count stream))
  (copy-device->device [_ dev-a dev-a-off dev-b dev-b-off elem-count]
    (tvm-shared/copy-device->device dev-a dev-a-off
                                    dev-b dev-b-off elem-count stream))
  (sync-with-host [_]
    (tvm-core/sync-stream-with-host (tvm-reg/device-type device)
                                    (tvm-reg/device-id device)
                                    stream))
  (sync-with-stream [src-stream dst-stream]
    (when-not (= (tvm-reg/device-type src-stream) (tvm-reg/device-type dst-stream))
      (throw (ex-info "Cannot synchronize streams of two different device types"
                      {:src-device-type (tvm-core/device-type-int->device-type
                                         (tvm-reg/device-type src-stream))
                       :dst-device-type (tvm-core/device-type-int->device-type
                                         (tvm-reg/device-type dst-stream))})))
    (tvm-core/sync-stream-with-stream (tvm-reg/device-type device)
                                      (tvm-reg/device-id device)
                                      (tvm-base/->tvm src-stream)
                                      (tvm-base/->tvm dst-stream)))
  tvm-reg/PTVMStream
  (call-function-impl [_ fn arg-list]
    (when (.address (tvm-base/->tvm stream))
      (tvm-core/set-current-thread-stream (tvm-reg/device-type device)
                                          (tvm-reg/device-id device)
                                          stream))
    (apply fn arg-list))

  drv/PDeviceProvider
  (get-device [_] device)

  drv/PDriverProvider
  (get-driver [_] (drv/get-driver device))

  resource/PResource
  (release-resource [_] ))


(defrecord GPUDevice [driver ^long device-id supports-create? default-stream resource-context]
  tvm-reg/PDriverInfo
  (device-type [this] (:device-type driver))

  tvm-reg/PDeviceInfo
  (device-id [this] device-id)

  drv/PDevice
  (memory-info-impl [device]
    ;;This would be useful information
    {:free 0xFFFFFFFF
     :total 0xFFFFFFF})
  (create-stream-impl [device]
    (->GPUStream device (tvm-core/create-stream (:device-type driver) device-id)))
  (allocate-device-buffer-impl [device elem-count elem-type]
    (dbuf/make-device-buffer-of-type device elem-type elem-count))
  (allocate-rand-buffer-impl [device elem-count]
    (dbuf/make-device-buffer-of-type device :float32 elem-count))
  (supports-create-stream? [device] supports-create?)
  (default-stream [device] @default-stream)

  drv/PDriverProvider
  (get-driver [dev] driver)

  drv/PDeviceProvider
  (get-device [dev] dev)

  resource/PResource
  (release-resource [_] (resource/release-resource resource-context)))


(defn- make-gpu-device
  "Never call this external; devices are centrally created and registered."
  [driver dev-id]
  (let [dev-type (:device-type driver)
        [default-stream resource-context]
        (resource/return-resource-context
         (try
           (tvm-core/create-stream dev-type dev-id)
           (catch Throwable e
             (tvm-base/->StreamHandle dev-type dev-id (runtime$TVMStreamHandle.)))))
        supports-create? (boolean default-stream)
        device (->GPUDevice driver dev-id supports-create? (atom nil) resource-context)]
    (swap! (:default-stream device) (constantly (->GPUStream device default-stream)))
    device))


(def ^:private enumerate-devices
  (memoize
   (fn [driver]
     (->> (tvm-shared/enumerate-devices (:device-type driver))
          (mapv #(make-gpu-device driver %))))))


(defrecord GPUDriver [^long device-type]
  tvm-reg/PDriverInfo
  (device-type [this] device-type)

  drv/PDriver
  (get-devices [driver]
    (enumerate-devices driver))
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    (dbuf/make-cpu-device-buffer elem-type elem-count))

  tvm-reg/PDeviceIdToDevice
  (device-id->device [driver dev-id]
    (if-let [retval (nth (drv/get-devices driver) dev-id)]
      retval
      (throw (ex-info "Device does not exist"
                      {:device-type (tvm-core/device-type-int->device-type device-type)
                       :device-id dev-id}))))

  tvm-reg/PCompileModule
  (gpu-scheduling? [driver] true)
  (device-datatypes? [driver] true)
  (schedule-injective! [driver stage compute-op {:keys [thread-count]}]
    ;;TODO - query device api for details like max thread count
    (let [device-max-thread-count 16]
      (api/stage-gpu-injective stage compute-op
                               :thread-count (or thread-count
                                                 device-max-thread-count))))

  (->module-impl [driver sched-data-seq options]
    (api/schedules->fns sched-data-seq
                        :build-config (:build-config options)
                        :target-host (:target-host options)
                        :target-name (tvm-core/device-type-int->device-type device-type))))


(def gpu-device-types #{:cuda :opencl :rocm})


(def driver
  (memoize
   (fn [device-type]
     (let [device-type (long (if (number? device-type)
                               device-type
                               (tvm-core/device-type->device-type-int device-type)))]
       (when-not (gpu-device-types (tvm-core/device-type-int->device-type device-type))
         (throw (ex-info "Device type does not appear to be a gpu device"
                         {:device-type (tvm-core/device-type-int->device-type
                                        device-type)})))
       (->GPUDriver device-type)))))


(defn cuda-driver [] (driver :cuda))
(defn opencl-driver [] (driver :opencl))
(defn rocm-driver [] (driver :rocm))


(doseq [dev-type gpu-device-types]
  (tvm-reg/add-device-type (tvm-core/device-type->device-type-int dev-type)
                                 (driver dev-type)))
