(ns tvm-clj.compute.device-buffer
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tvm-clj.compute.base :as tvm-comp-base]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [tvm-clj.compute.host-buffer :as hbuf]
            [think.resource.core :as resource]
            [clojure.core.matrix.protocols :as mp]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.datatype.marshal :as marshal])
  (:import [tvm_clj.tvm runtime$DLTensor runtime]
           [tvm_clj.base ArrayHandle]
           [org.bytedeco.javacpp Pointer LongPointer]
           [java.lang.reflect Field]))


(defn base-ptr-dtype
  [datatype]
  (if (hbuf/signed-datatype? datatype)
    datatype
    (hbuf/direct-conversion-map datatype)))

(defn tvm-ary->pointer
  ^Pointer [^ArrayHandle ten-ary ^long elem-count datatype]
  (let [^runtime$DLTensor tensor (.tvm-jcpp-handle ten-ary)
        tens-ptr (.data tensor)
        ptr-dtype (base-ptr-dtype datatype)
        retval (jcpp-dtype/make-empty-pointer-of-type ptr-dtype)]
    (.set ^Field jcpp-dtype/address-field retval (+ (.address tens-ptr)
                                                    (.byte_offset tensor)))
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
  (* 8 (dtype/datatype->byte-size datatype)))

(defn pointer->tvm-ary
  "Not all backends in TVM can offset their pointer types.  For this reason, tvm arrays
have a byte_offset member that you can use to make an array not start at the pointer's
base address."
  ^ArrayHandle [^Pointer ptr device-type device-id datatype elem-count byte-offset]
  (let [tens-data (runtime$DLTensor. 1)
        ctx (.ctx tens-data)
        dtype (.dtype tens-data)
        shape (LongPointer. 1)
        elem-count (long elem-count)]
    (.put shape 0 elem-count)
    (.data tens-data ptr)
    (.ndim tens-data 1)
    (.byte_offset tens-data (long byte-offset))
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


(defn is-cpu-device?
  [device]
  (= runtime/kDLCPU (tvm-comp-base/device-type device)))


(declare device-buffer->ptr)


(defrecord DeviceBuffer [device dev-ary]
  dtype/PDatatype
  (get-datatype [buf] (:datatype dev-ary))

  mp/PElementCount
  (element-count [buf] (apply * (:shape dev-ary)))

  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (let [base-ptr (device-buffer->ptr buffer)
          datatype (dtype/get-datatype buffer)]
      (->DeviceBuffer device
                      (pointer->tvm-ary base-ptr
                                        (tvm-comp-base/device-type device)
                                        (tvm-comp-base/device-id device)
                                        datatype
                                        length
                                        ;;add the byte offset where the new pointer should start
                                        (* (long offset) (long (dtype/datatype->byte-size
                                                                datatype)))))))
  (alias? [lhs rhs]
    (hbuf/jcpp-pointer-alias? (tvm-ary->pointer dev-ary)
                              (:dev-ary rhs)))
  (partially-alias? [lhs rhs]
    (hbuf/jcpp-pointer-partial-alias? (tvm-ary->pointer dev-ary)
                                      (tvm-ary->pointer (:dev-ary rhs))))

  tvm-comp-base/PConvertToTVM
  (->tvm [item] dev-ary)


  dtype/PAccess
  (set-value! [item offset value]
    (let [conv-fn (get-in hbuf/unsigned-scalar-conversion-table [(dtype/get-datatype item) :to])
          value (if conv-fn (conv-fn value) value)]
      (dtype/set-value! (device-buffer->ptr item) offset value)))
  (set-constant! [item offset value elem-count]
    (let [conv-fn (get-in hbuf/unsigned-scalar-conversion-table [(dtype/get-datatype item) :to])
          value (if conv-fn (conv-fn value) value)]
      (dtype/set-constant! (device-buffer->ptr item) offset value elem-count)))
  (get-value [item offset]
    (let [conv-fn (get-in hbuf/unsigned-scalar-conversion-table
                          [(dtype/get-datatype item) :from])
          ptr (device-buffer->ptr item)]
      (if conv-fn
        (conv-fn (dtype/get-value ptr offset))
        (dtype/get-value ptr offset))))

  hbuf/PToPtr
  (->ptr [item] (device-buffer->ptr item))

  ;;Efficient bulk copy is provided by this line and implementing the PToPtr protocol
  marshal/PContainerType
  (container-type [this] :tvm-host-buffer))


(defn device-buffer->ptr
  "Get a javacpp pointer from a device buffer.  Throws if this isn't a cpu buffer"
  [^DeviceBuffer buffer]
  (when-not (is-cpu-device? (.device buffer))
    (throw (ex-info "Can only get pointers from cpu device buffers")))
  (tvm-ary->pointer (.dev-ary buffer) (mp/element-count buffer) (dtype/get-datatype buffer)))


(defn make-device-buffer-of-type
  [device datatype elem-count]
  (->> (tvm-core/allocate-device-array [elem-count] datatype
                                      (tvm-comp-base/device-type device)
                                      (tvm-comp-base/device-id device))
       (->DeviceBuffer device)))




(defn make-cpu-device-buffer
  "The cpu driver needs to be required for this to work."
  [datatype elem-count]
  (when-not (resolve 'tvm-clj.compute.cpu/cpu-driver)
    (require 'tvm-clj.compute.cpu))
  (make-device-buffer-of-type (tvm-comp-base/get-device runtime/kDLCPU 0)
                              datatype
                              elem-count))


(defn device-buffer->tvm-array
  ^runtime$DLTensor [^DeviceBuffer buf]
  (.tvm-jcpp-handle (.dev-ary buf)))
