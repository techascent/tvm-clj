(ns tech.compute.tvm.device-buffer
  (:require [tvm-clj.tvm-jna :as bindings]
            [tvm-clj.bindings.protocols :as tvm-proto]
            [tech.compute.driver :as drv]
            [tech.v2.datatype :as dtype]
            [tech.compute :as compute])
  (:import  [tvm_clj.tvm DLPack$DLTensor]
            [com.sun.jna Pointer]))

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
  drv/PBuffer
  (sub-buffer [buffer offset length]
    ;;We don't use the PToPtr protocol because we do actually need the base ptr address.
    ;;->ptr-backing-store offsets that address.
    (let [base-ptr (bindings/base-ptr buffer)
          datatype (dtype/get-datatype buffer)
          byte-offset (long (bindings/byte-offset buffer))]
      (bindings/pointer->tvm-ary base-ptr
                                 (bindings/device-type buffer)
                                 (bindings/device-id buffer)
                                 datatype
                                 [length]
                                 nil
                                 ;;add the byte offset where the new pointer should start
                                 (+ byte-offset
                                    (* (long offset)
                                       (long (dtype/datatype->byte-size
                                              datatype))))
                                 ;;Use the passed in buffer as a gc-root object; obviously the new thing
                                 ;;should refer back to the source.
                                 buffer)))
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
  tvm-proto/PToTVM
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
      (if (and (nil? stride-data)
               (= (dtype/shape buffer)
                  (:shape dims)))
        buffer
        (bindings/pointer->tvm-ary (bindings/base-ptr buffer)
                                   (bindings/device-type buffer)
                                   (bindings/device-id buffer)
                                   (ct/get-datatype item)
                                   (:shape dims)
                                   stride-data
                                   (bindings/byte-offset buffer)))))

  tvm-proto/PJVMTypeToTVMValue
  (->tvm-value [item]
    (-> (bindings/->tvm item)
        bindings/->tvm-value))

  tvm-proto/PByteOffset
  (byte-offset [tensor]
    (bindings/byte-offset (ct/tensor->buffer tensor)))
  (base-ptr [tensor]
    (bindings/base-ptr (ct/tensor->buffer tensor)))
  )
