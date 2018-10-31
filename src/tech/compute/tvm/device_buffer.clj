(ns tech.compute.tvm.device-buffer
  (:require [tvm-clj.tvm-jna :as bindings]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.datatype.base :as dtype-base]
            [tech.datatype :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.resource :as resource]
            [clojure.core.matrix.protocols :as mp]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [tech.compute :as compute]
            [tech.compute.tvm :as compute-tvm]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.datatype.jna :as dtype-jna])
  (:import  [tvm_clj.tvm DLPack$DLTensor]
            [com.sun.jna Pointer]
            [tech.compute.tensor Tensor]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn is-cpu-device?
  [device]
  (= :cpu (compute-tvm/device-type device)))


(defn check-cpu-array!
  [array]
  (when-not (= :cpu (bindings/device-type array))
    (throw (ex-info "Illegal operation on a non-cpu array."
                    {:device-type (bindings/device-type array)}))))


(extend-type DLPack$DLTensor
  drv/PBuffer
  (sub-buffer [buffer offset length]
    (let [base-ptr (dtype-jna/->ptr-backing-store buffer)
          datatype (dtype/get-datatype buffer)
          byte-offset (long (bindings/byte-offset buffer))]
      (bindings/pointer->tvm-ary
       base-ptr
       (long (bindings/device-type->device-type-int
              (compute-tvm/device-type buffer)))
       (long (compute-tvm/device-id buffer))
       datatype
       [length]
       nil
       ;;add the byte offset where the new pointer should start
       (+ byte-offset
          (* (long offset)
             (long (dtype/datatype->byte-size
                    datatype)))))))
  (alias? [lhs rhs]
    (drv/alias? (dtype-jna/->typed-pointer lhs) rhs))

  (partially-alias? [lhs rhs]
    (drv/partially-alias? (dtype-jna/->typed-pointer lhs) rhs))

  drv/PDeviceProvider
  (get-device [buffer]
    (-> (compute/->driver buffer)
        (compute-tvm/device-id->device
         (bindings/device-id buffer))))

  drv/PDriverProvider
  (get-driver [buffer]
    (-> (bindings/device-type buffer)
        compute-tvm/driver)))


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
                     (drv/sub-buffer src-buffer src-offset elem-count)
                     src-buffer)
        dst-buffer (if-not (and (= 0 (long dst-offset))
                                (= elem-count (dtype/ecount dst-buffer)))
                     (drv/sub-buffer dst-buffer dst-offset elem-count)
                     dst-buffer)]
    (bindings/copy-array-to-array! src-buffer dst-buffer stream)))


(extend-type Tensor
  bindings/PToTVM
  (->tvm [item]
    ;;This is a specialized conversion because the tensor's dimension change independent
    ;;of the buffer.  Thus any time we want to use a tensor in tvm we have to create an
    ;;alias of the base buffer but with variables set describing the current dimensions.
    (let [buffer (bindings/->tvm (ct/tensor->buffer item))
          dims (ct/tensor->dimensions item)
          stride-data (when-not (and (ct/dense? item)
                                     (ct-dims/access-increasing?
                                      (ct/tensor->dimensions item)))
                        (:strides dims))]
      (bindings/pointer->tvm-ary (dtype-jna/->ptr-backing-store buffer)
                                 (bindings/device-type buffer)
                                 (bindings/device-id buffer)
                                 (ct/get-datatype item)
                                 (:shape dims)
                                 stride-data
                                 (bindings/byte-offset buffer))))

  bindings/PJVMTypeToTVMValue
  (->tvm-value [item]
    (-> (bindings/->tvm item)
        bindings/->tvm-value))

  bindings/PByteOffset
  (byte-offset [tensor]
    (bindings/byte-offset (ct/tensor->buffer tensor))))
