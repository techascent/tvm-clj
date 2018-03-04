(ns tech.compute.tvm.device-buffer
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.compute.tvm.base :as tvm-comp-base]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [tech.compute.tvm.host-buffer :as hbuf]
            [think.resource.core :as resource]
            [clojure.core.matrix.protocols :as mp]
            [tech.javacpp-datatype :as jcpp-dtype])
  (:import [tvm_clj.tvm runtime$DLTensor runtime]
           [tvm_clj.base ArrayHandle]
           [org.bytedeco.javacpp Pointer]
           [java.lang.reflect Field]))


(defn base-ptr-dtype
  [datatype]
  (if (hbuf/signed-datatype? datatype)
    datatype
    (hbuf/direct-conversion-map datatype)))

(defn tvm-ary->pointer
  ^Pointer [^runtime$DLTensor tensor ^long elem-count datatype]
  (let [tens-ptr (.data tensor)
        ptr-dtype (base-ptr-dtype datatype)
        retval (jcpp-dtype/make-empty-pointer-of-type ptr-dtype)]
    (.set ^Field jcpp-dtype/address-field retval (.address tens-ptr))
    (jcpp-dtype/set-pointer-limit-and-capacity retval elem-count)))


(defn datatype->dl-type-code
  ^long [datatype]
  (condp = datatype
    :uint8 runtime/kDLUInt
    :uint16 runtime/kDLUInt
    :uint32 runtime/kDLUInt
    :uint64 runtime/kDLUInt
    :int8 runtime/kDLInt
    :int16 runtime/kDLInt
    :int32 runtime/kDLInt
    :int64 runtime/kDLInt
    :float32 runtime/kDLFloat
    :float64 runtime/kDLFloat))


(defn datatype->dl-bits
  ^long [datatype]
  (condp = datatype
    :uint8 8
    :uint16 16
    :uint32 32
    :uint64 64
    :int8 8
    :int16 16
    :int32 32
    :int64 64
    :float32 32
    :float64 64))

(defn pointer->tvm-ary
  ^ArrayHandle [^Pointer ptr device-type device-id datatype]
  (let [tens-data (runtime$DLTensor. 1)
        num-items (dtype/ecount ptr)
        ctx (.ctx tens-data)
        dtype (.dtype tens-data)
        shape (.LongPointer 1)
        elem-count (long (mp/element-count ptr))]
    (.set shape 0 elem-count)
    (.data tens-data ptr)
    (.ndims tens-data 1)
    (.device_type ctx (int device-type))
    (.device_id ctx (int device-id))
    (.code dtype (datatype->dl-type-code datatype))
    (.bits dtype (datatype->dl-bits datatype))
    (.lanes dtype 1)
    (.shape tens-data shape)
    (resource/track shape)
    (resource/track
     (merge (tvm-base/->ArrayHandle tens-data)
            {:shape [elem-count]
             :datatype datatype
             :owns-memory? false}))))


(defrecord DeviceBuffer [device dev-ary]
  dtype/PDatatype
  (get-datatype [buf] (:datatype dev-ary))

  mp/PElementCount
  (element-count [buf] (apply * (:shape dev-ary)))

  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (let [base-ptr (tvm-ary->pointer dev-ary)
          new-ptr (hbuf/jcpp-pointer-sub-buffer base-ptr offset length)]
      (->DeviceBuffer device
                      (pointer->tvm-ary new-ptr
                                        (tvm-comp-base/device-type device)
                                        (tvm-comp-base/device-id device)
                                        (dtype/get-datatype buffer)))))
  (alias? [lhs rhs]
    (hbuf/jcpp-pointer-alias? (tvm-ary->pointer dev-ary)
                              (:dev-ary rhs)))
  (partially-alias? [lhs rhs]
    (hbuf/jcpp-pointer-partial-alias? (tvm-ary->pointer dev-ary)
                                      (tvm-ary->pointer (:dev-ary rhs))))

  tvm-comp-base/PConvertToTVM
  (->tvm [item] dev-ary))


(defn make-device-buffer-of-type
  [device datatype elem-count]
  (->> (tvm-core/allocate-device-array [elem-count] datatype
                                      (tvm-comp-base/device-type device)
                                      (tvm-comp-base/device-id device))
       (->DeviceBuffer device)))


(defn device-buffer->tvm-array
  ^runtime$DLTensor [^DeviceBuffer buf]
  (.tvm-jcpp-handle (.dev-ary buf)))
