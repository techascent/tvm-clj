(ns tvm-clj.compute.gpu
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.compute.driver :as drv]
            [tvm-clj.compute.base :as tvm-comp-base]
            [tech.datatype.base :as dtype]
            [tvm-clj.compute.device-buffer :as dbuf]
            [tvm-clj.compute.shared :as tvm-shared]
            [think.resource.core :as resource]
            ))


(declare cuda-driver)
(declare opencl-driver)
(declare rocm-driver)
(declare make-gpu-device)


(defrecord GPUStream [device stream]
  tvm-base/PToTVM
  (->tvm [_] stream)

  tvm-comp-base/PDeviceInfo
  (device-type [_] (tvm-comp-base/device-type device))
  (device-id [_] (tvm-comp-base/device-id device))

  drv/PStream
  (copy-host->device [stream host-buffer host-offset
                      device-buffer device-offset elem-count]
    (tvm-shared/copy-device->device host-buffer host-offset
                                    device-buffer device-offset elem-count stream))
  (copy-device->host [stream device-buffer device-offset
                      host-buffer host-offset elem-count]
    (tvm-shared/copy-device->device device-buffer device-offset
                                    host-buffer host-offset elem-count stream))
  (copy-device->device [stream dev-a dev-a-off dev-b dev-b-off elem-count]
    (tvm-shared/copy-device->device dev-a dev-a-off
                                    dev-b dev-b-off elem-count stream))
  (memset [stream device-buffer device-offset elem-val elem-count]
    (throw (ex-info "Not implemented yet.")))
  (sync-with-host [_]
    (tvm-core/sync-stream-with-host (tvm-comp-base/device-type device)
                                    (tvm-comp-base/device-id device)
                                    stream))
  (sync-with-stream [src-stream dst-stream]
    (when-not (= (tvm-comp-base/device-type src-stream) (tvm-comp-base/device-type dst-stream))
      (throw (ex-info "Cannot synchronize streams of two different device types"
                      {:src-device-type (tvm-core/device-type-int->device-type (tvm-comp-base/device-type src-stream))
                       :dst-device-type (tvm-core/device-type-int->device-type (tvm-comp-base/device-type dst-stream))})))
    (tvm-core/sync-stream-with-stream (tvm-comp-base/device-type device)
                                      (tvm-comp-base/device-id device)
                                      (tvm-base/->tvm src-stream)
                                      (tvm-base/->tvm dst-stream)))
  resource/PResource
  (release-resource [_] ))


(defrecord GPUDevice [driver ^long device-id]
  tvm-comp-base/PDeviceInfo
  (device-type [this] (:device-type driver))
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
    (dbuf/make-device-buffer-of-type device :float32 elem-count)))

(defn make-gpu-device [driver dev-id]
  (->GPUDevice driver dev-id))


(def ^:private enumerate-devices
  (memoize
   (fn [driver]
     (->> (tvm-shared/enumerate-devices (:device-type driver))
          (mapv #(make-gpu-device driver %))))))


(defrecord GPUDriver [^long device-type]
  drv/PDriver
  (get-devices [driver]
    (enumerate-devices driver))
  (allocate-host-buffer-impl [driver elem-count elem-type options]
    (dbuf/make-cpu-device-buffer elem-type elem-count))

  tvm-comp-base/PDeviceIdToDevice
  (device-id->device [driver dev-id]
    (if-let [retval (nth (drv/get-devices driver) dev-id)]
      retval
      (throw (ex-info "Device does not exist"
                      {:device-type (tvm-core/device-type-int->device-type device-type)
                       :device-id dev-id})))))


(def gpu-device-types #{:cuda :opencl :rocm})


(def driver
  (memoize
   (fn [device-type]
     (let [device-type (long (if (number? device-type)
                               device-type
                               (tvm-core/device-type->device-type-int device-type)))]
       (when-not (gpu-device-types (tvm-core/device-type-int->device-type device-type))
         (throw (ex-info "Device type does not appear to be a gpu device"
                         {:device-type (tvm-core/device-type-int->device-type device-type)})))
       (->GPUDriver device-type)))))


(defn cuda-driver [] (driver :cuda))
(defn opencl-driver [] (driver :opencl))
(defn rocm-driver [] (driver :rocm))


(doseq [dev-type gpu-device-types]
  (tvm-comp-base/add-device-type (tvm-core/device-type->device-type-int dev-type)
                                 (driver dev-type)))
