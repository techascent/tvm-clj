(ns tvm-clj.device
  "Operations on a device.  Devices, such as a GPU, need to be addressed
  independently and once you have a device you can allocate a tensor on that
  device.

  * Device types are keywords: `#{:cpu :cuda :opencl}`
  * Device ids are integers starting from zero."
  (:require [tvm-clj.impl.definitions :as definitions]
            [tvm-clj.impl.protocols :as tvm-proto]
            [tvm-clj.impl.fns.runtime :as runtime-fns]
            [tvm-clj.impl.dl-tensor :as dl-tensor]
            [tvm-clj.impl.stream :as stream]
            [tvm-clj.impl.base :as base]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]))


(defn device-exists?
  [device-type device-id]
  (if (= device-type :cpu)
    (== 0 (long device-id)))
  (= 1 (runtime-fns/GetDeviceAttr
        (definitions/device-type->device-type-int device-type)
        (long device-id) (definitions/device-attribute-map :exists))))


(defn device-attributes
  [device-type device-id]
  (when (device-exists? device-type device-id)
    (->> definitions/device-attribute-map
         (map (fn [[att-name att-id]]
                [att-name (runtime-fns/GetDeviceAttr
                           (definitions/device-type->device-type-int device-type)
                           (long device-id)
                           (long att-id))]))
         (into {}))))


(defn device-tensor
  "Allocate a device tensor."
  ([shape datatype device-type device-id options]
   (dl-tensor/allocate-device-array shape datatype device-type
                                    device-id options))
  ([shape datatype device-type device-id]
   (device-tensor shape datatype device-type device-id nil))
  ([src-tens-prototype device-type device-id]
   (device-tensor (dtype/shape src-tens-prototype)
                  (dtype/elemwise-datatype src-tens-prototype)
                  device-type device-id nil)))


(defn copy-tensor!
  "Copy a src tensor to a destination tensor."
  ([src-tens dest-tens stream]
   (dl-tensor/copy-array-to-array! src-tens dest-tens stream)
   dest-tens)
  ([src-tens dest-tens]
   (copy-tensor! src-tens dest-tens nil)))


(defn sync-with-host
  "Synchonize the device stream with the host"
  [device-type device-id]
  (base/check-call (stream/TVMSynchronize device-type device-id nil)))


(defn cpu->device
  "Ensure a tensor is on a device copying if necessary."
  ([tensor device-type device-id {:keys [stream] :as options}]
   (let [dev-tens (device-tensor (dtype/shape tensor)
                                 (dtype/elemwise-datatype tensor)
                                 device-type device-id options)
         ;;This will make a gc-based tensor so be careful.
         tensor (if (dtt/dims-suitable-for-desc? tensor)
                  tensor
                  (dtt/clone tensor :container-type :native-heap))]
     (copy-tensor! tensor dev-tens stream)))
  ([tensor device-type device-id]
   (cpu->device tensor device-type device-id nil)))


(defn device->cpu
  "Ensure a tensor is on a device copying if necessary."
  ([dev-tens {:keys [stream unsynchronized?]}]
   (let [tensor (dtt/new-tensor (dtype/shape dev-tens)
                                :datatype (dtype/elemwise-datatype dev-tens)
                                :container-type :native-heap)]
     (copy-tensor! dev-tens tensor stream)
     (when-not unsynchronized?
       (sync-with-host (tvm-proto/device-type dev-tens)
                       (tvm-proto/device-id dev-tens)))
     tensor))
  ([dev-tens]
   (device->cpu dev-tens nil)))
