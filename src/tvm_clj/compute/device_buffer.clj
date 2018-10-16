(ns tvm-clj.compute.device-buffer
  (:require [tvm-clj.tvm-bindings :as bindings]
            [tvm-clj.base :as tvm-base]
            [tvm-clj.compute.registry :as tvm-reg]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.core :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.resource :as resource]
            [clojure.core.matrix.protocols :as mp]
            [tech.datatype.javacpp :as jcpp-dtype]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.tensor :as ct])
  (:import [tvm_clj.tvm runtime$DLTensor runtime runtime$DLContext]
           [tvm_clj.base ArrayHandle]
           [org.bytedeco.javacpp Pointer LongPointer]
           [java.lang.reflect Field]
           [tech.compute.tensor Tensor]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn is-cpu-device?
  [device]
  (= runtime/kDLCPU (tvm-reg/device-type device)))


(declare device-buffer->ptr)


(defn jcpp-pointer-alias?
  [^Pointer lhs ^Pointer rhs]
  (= (.address lhs)
     (.address rhs)))

(defn jcpp-pointer-byte-length
  ^long [^Pointer ptr]
  (long (* (dtype/ecount ptr)
           (dtype/datatype->byte-size
            (dtype/get-datatype ptr)))))


(defn jcpp-pointer-partial-alias?
  [^Pointer lhs ^Pointer rhs]
  (let [l-start (.address lhs)
        r-start (.address rhs)
        l-end (+ l-start (jcpp-pointer-byte-length lhs))
        r-end (+ r-start (jcpp-pointer-byte-length rhs))]
    (or (and (>= r-start l-start)
             (< r-start l-end))
        (and (>= l-start r-start)
             (< l-start r-end)))))


(defn check-cpu-array!
  [^ArrayHandle array]
  (when-not (= runtime/kDLCPU (long (tvm-reg/device-type array)))
    (throw (ex-info "Illegal operation on a non-cpu array."
                    {:device-type (tvm-reg/device-type-int->keyword
                                   (long (tvm-reg/device-type array)))}))))


(extend-type ArrayHandle
  dtype-base/PDatatype
  (get-datatype [buf] (:datatype buf))

  dtype-base/PAccess
  (set-value! [item offset value]
    (dtype-base/set-value! (unsigned/->typed-buffer item)
                           offset value))
  (set-constant! [item offset value elem-count]
    (dtype-base/set-constant! (unsigned/->typed-buffer item)
                              offset value elem-count))
  (get-value [item offset]
    (dtype-base/get-value (unsigned/->typed-buffer item)
                          offset))

  dtype-base/PContainerType
  (container-type [item] :typed-buffer)

  dtype-base/PCopyRawData
  (copy-raw->item! [raw-data ary-target target-offset options]
    (dtype-base/copy-raw->item! (unsigned/->typed-buffer raw-data) ary-target
                                target-offset options))

  mp/PElementCount
  (element-count [buf] (apply * (mp/get-shape buf)))

  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (:shape m))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))


  jcpp-dtype/PToPtr
  (->ptr-backing-store [item]
    (tvm-base/tvm-ary->pointer item
                               (dtype/ecount item)
                               (dtype/get-datatype item)))


  primitive/PToBuffer
  (->buffer-backing-store [item]
    (check-cpu-array! item)
    (jcpp-dtype/ptr->buffer (jcpp-dtype/->ptr-backing-store item)))


  primitive/PToArray
  (->array [item] nil)
  (->array-copy [item]
    (check-cpu-array! item)
    (primitive/->array-copy
     (unsigned/->typed-buffer item)))


  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (let [^runtime$DLTensor tvm-tensor (tvm-base/->tvm buffer)
          base-ptr (.data tvm-tensor)
          datatype (dtype/get-datatype buffer)]
      (tvm-base/pointer->tvm-ary
       base-ptr
       (long (tvm-reg/device-type buffer))
       (long (tvm-reg/device-id buffer))
       datatype
       [length]
       nil
       ;;add the byte offset where the new pointer should start
       (* (long offset) (long (dtype/datatype->byte-size
                               datatype))))))
  (alias? [lhs rhs]
    (jcpp-pointer-alias? (jcpp-dtype/->ptr-backing-store lhs)
                         (jcpp-dtype/->ptr-backing-store rhs)))
  (partially-alias? [lhs rhs]
    (jcpp-pointer-partial-alias? (jcpp-dtype/->ptr-backing-store lhs)
                                 (jcpp-dtype/->ptr-backing-store rhs)))

  tvm-reg/PDeviceInfo
  (device-id [buffer]
    (let [^runtime$DLTensor tensor (tvm-base/->tvm buffer)
          ctx (.ctx tensor)]
      (.device_id ctx)))

  tvm-reg/PDriverInfo
  (device-type [buffer]
    (let [^runtime$DLTensor tensor (tvm-base/->tvm buffer)
          ctx (.ctx tensor)]
      (.device_type ctx)))

  drv/PDeviceProvider
  (get-device [buffer]
    (tvm-reg/get-device (tvm-reg/device-type buffer)
                        (tvm-reg/device-id buffer)))

  drv/PDriverProvider
  (get-driver [buffer]
    (tvm-reg/get-driver (tvm-reg/device-type buffer))))


(defn make-device-buffer-of-type
  [device datatype elem-count]
  (bindings/allocate-device-array [elem-count] datatype
                                  (tvm-reg/device-type device)
                                  (tvm-reg/device-id device)))


(defn make-cpu-device-buffer
  "Make a cpu device buffer.  These are used as host buffers for all devices
and device buffers for the cpu device."
  [datatype elem-count]
  (when-not (resolve 'tvm-clj.compute.cpu/driver)
    (require 'tvm-clj.compute.cpu))
  (make-device-buffer-of-type (tvm-reg/get-device runtime/kDLCPU 0)
                              datatype
                              elem-count))


(defn device-buffer->dl-tensor
  ^runtime$DLTensor [^ArrayHandle buf]
  (tvm-base/->tvm buf))


(defn has-byte-offset?
  [tensor]
  (let [buf-data (ct/tensor->buffer tensor)
        ^runtime$DLTensor backing-store (tvm-base/->tvm buf-data)]
    (not= 0 (.byte_offset backing-store))))


(extend-type Tensor
  tvm-base/PToTVM
  (->tvm [item]
    ;;This is a specialized conversion because the tensor's dimension change independent
    ;;of the buffer.  Thus any time we want to use a tensor in tvm we have to create
    ;;an alias of the base buffer but with variables set describing the current
    ;;dimensions.
    (let [^runtime$DLTensor src-dl-tensor (tvm-base/->tvm (ct/tensor->buffer item))
          ^runtime$DLContext ctx (.ctx src-dl-tensor)
          dims (ct/tensor->dimensions item)
          stride-data (when-not (ct/dense? item)
                        (:strides dims))]

      (tvm-base/pointer->tvm-ary (.data src-dl-tensor)
                                 (.device_type ctx)
                                 (.device_id ctx)
                                 (ct/get-datatype item)
                                 (:shape dims)
                                 stride-data
                                 (.byte_offset src-dl-tensor))))
  tvm-base/PJVMTypeToTVMValue
  (->tvm-value [item]
    (-> (tvm-base/->tvm item)
        tvm-base/->tvm-value)))
