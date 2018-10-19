(ns tech.compute.tvm.device-buffer
  (:require [tvm-clj.tvm-bindings :as bindings]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.core :as dtype]
            [tech.datatype.java-primitive :as primitive]
            [tech.resource :as resource]
            [clojure.core.matrix.protocols :as mp]
            [tech.datatype.javacpp :as jcpp-dtype]
            [tech.datatype.java-unsigned :as unsigned]
            [tech.compute.tensor :as ct]
            [tech.compute :as compute]
            [tech.compute.tvm :as compute-tvm]
            [tech.compute.tvm.driver :as tvm-driver])
  (:import [tvm_clj.tvm runtime$DLTensor runtime runtime$DLContext]
           [tvm_clj.tvm_bindings ArrayHandle]
           [org.bytedeco.javacpp Pointer LongPointer]
           [java.lang.reflect Field]
           [tech.compute.tensor Tensor]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn is-cpu-device?
  [device]
  (= :cpu (compute-tvm/device-type device)))


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
  (when-not (is-cpu-device? array)
    (throw (ex-info "Illegal operation on a non-cpu array."
                    {:device-type (-> array
                                      (compute/->driver)
                                      )}))))


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
    (bindings/tvm-ary->pointer item
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
  (sub-buffer [buffer offset length]
    (let [^runtime$DLTensor tvm-tensor (bindings/->tvm buffer)
          base-ptr (.data tvm-tensor)
          datatype (dtype/get-datatype buffer)]
      (bindings/pointer->tvm-ary
       base-ptr
       (long (bindings/device-type->device-type-int
              (compute-tvm/device-type buffer)))
       (long (compute-tvm/device-id buffer))
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

  tvm-driver/PTVMDeviceId
  (device-id [buffer]
    (let [^runtime$DLTensor tensor (bindings/->tvm buffer)
          ctx (.ctx tensor)]
      (.device_id ctx)))

  tvm-driver/PTVMDeviceType
  (device-type [buffer]
    (let [^runtime$DLTensor tensor (bindings/->tvm buffer)
          ctx (.ctx tensor)]
      (bindings/device-type-int->device-type
       (.device_type ctx))))

  tvm-driver/PTVMBuffer
  (has-byte-offset? [buffer]
    (let [^runtime$DLTensor backing-store (bindings/->tvm buffer)]
      (not= 0 (.byte_offset backing-store))))

  drv/PDeviceProvider
  (get-device [buffer]
    (-> (compute/->driver buffer)
        (compute-tvm/device-id->device
         (tvm-driver/device-id buffer))))

  drv/PDriverProvider
  (get-driver [buffer]
    (-> (tvm-driver/device-type buffer)
        compute-tvm/driver)))


(defn make-device-buffer-of-type
  [device datatype elem-count]
  (bindings/allocate-device-array [elem-count] datatype
                                  (compute-tvm/device-type device)
                                  (compute-tvm/device-id device)))



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
    ;;of the buffer.  Thus any time we want to use a tensor in tvm we have to create
    ;;an alias of the base buffer but with variables set describing the current
    ;;dimensions.
    (let [^runtime$DLTensor src-dl-tensor (bindings/->tvm (ct/tensor->buffer item))
          ^runtime$DLContext ctx (.ctx src-dl-tensor)
          dims (ct/tensor->dimensions item)
          stride-data (when-not (ct/dense? item)
                        (:strides dims))]

      (bindings/pointer->tvm-ary (.data src-dl-tensor)
                                 (.device_type ctx)
                                 (.device_id ctx)
                                 (ct/get-datatype item)
                                 (:shape dims)
                                 stride-data
                                 (.byte_offset src-dl-tensor))))
  bindings/PJVMTypeToTVMValue
  (->tvm-value [item]
    (-> (bindings/->tvm item)
        bindings/->tvm-value))

  tvm-driver/PTVMBuffer
  (has-byte-offset? [tensor]
    (tvm-driver/has-byte-offset? (ct/tensor->buffer tensor))))
