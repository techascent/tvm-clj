(ns tech.libs.tvm.device-buffer
  (:require [tvm-clj.tvm-jna :as bindings]
            [tvm-clj.bindings.protocols :as tvm-proto]
            [tech.compute.driver :as drv]
            [tech.libs.tvm.driver :as tvm-driver]
            [tech.libs.tvm :as tech-tvm]
            [tech.v2.datatype :as dtype]
            [tech.compute :as compute]
            [tech.v2.tensor :as dtt])
  (:import  [tvm_clj.tvm DLPack$DLTensor]
            [com.sun.jna Pointer]
            [tech.v2.tensor.impl Tensor]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn is-cpu-device?
  [device]
  (= :cpu (bindings/device-type device)))


(defn check-cpu-array!
  [array]
  (when-not (= :cpu (bindings/device-type array))
    (throw (ex-info "Illegal operation on a non-cpu array."
                    {:device-type (bindings/device-type array)}))))


(extend-type DLPack$DLTensor

  drv/PDeviceProvider
  (get-device [buffer]
    (-> (compute/->driver buffer)
        (tvm-driver/device-id->device
         (bindings/device-id buffer))))

  drv/PDriverProvider
  (get-driver [buffer]
    (-> (bindings/device-type buffer)
        tech-tvm/driver)))


(defn make-device-buffer-of-type
  [device datatype elem-count]
  (bindings/allocate-device-array [elem-count] datatype
                                  (bindings/device-type device)
                                  (bindings/device-id device)))


(defn copy-device->device
  [src-buffer src-offset dst-buffer dst-offset elem-count stream]
  (let [elem-count (long elem-count)
        src-buffer (if-not (and (= 0 (long src-offset))
                                (= elem-count (dtype/ecount src-buffer)))
                     (dtype/sub-buffer src-buffer src-offset elem-count)
                     src-buffer)
        dst-buffer (if-not (and (= 0 (long dst-offset))
                                (= elem-count (dtype/ecount dst-buffer)))
                     (dtype/sub-buffer dst-buffer dst-offset elem-count)
                     dst-buffer)]
    (bindings/copy-array-to-array! src-buffer dst-buffer stream)))


(extend-type Tensor
  tvm-proto/PJVMTypeToTVMValue
  (->tvm-value [item]
    (-> (bindings/->tvm item)
        bindings/->tvm-value))

  tvm-proto/PByteOffset
  (byte-offset [tensor]
    (bindings/byte-offset (dtt/tensor->buffer tensor)))
  (base-ptr [tensor]
    (bindings/base-ptr (dtt/tensor->buffer tensor))))
