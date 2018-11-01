(ns tvm-clj.jna.dl-tensor
  (:require [tvm-clj.jna.base :refer [make-tvm-jna-fn
                                      device-type->int
                                      device-id->int
                                      ptr-ptr
                                      check-call
                                      ->long-ptr
                                      datatype->dl-datatype
                                      dl-datatype->datatype]]
            [tvm-clj.jna.stream :as stream]
            [tech.resource :as resource]
            [tvm-clj.bindings.protocols :refer [->tvm
                                                base-ptr
                                                ->tvm-value
                                                byte-offset] :as bindings-proto]
            [tech.datatype.jna :as dtype-jna]
            [tvm-clj.bindings.definitions :refer [device-type-int->device-type]]

            [tech.jna :refer [checknil] :as jna]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.protocols :as mp]
            [tech.datatype :as dtype]
            )
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]))


(defn ensure-array
  [item]
  (when-not (or (instance? DLPack$DLTensor item)
                (instance? DLPack$DLManagedTensor item))
    (throw (ex-info "Item is neither tensor or managed tensor"
                    {:item item
                     :item-type (type item)})))
  item)


;; Arrays
(make-tvm-jna-fn TVMArrayFree
                 "Free a TVM array allocated with TVMArrayAlloc"
                 Integer
                 [item ensure-array])


(defn check-cpu-tensor
  [^DLPack$DLTensor item]
  (let [item-device (-> item
                        (.ctx)
                        (.device_type)
                        device-type-int->device-type)]
    (when-not
        (= :cpu item-device)
      (throw (ex-info "Item is not a cpu tensor"
                      {:item-device item-device})))
    item))


(extend-type DLPack$DLTensor
  bindings-proto/PToTVM
  (->tvm [item] item)
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [item]
    (.write item)
    [(-> (.getPointer item)
         Pointer/nativeValue)
     :array-handle])
  dtype-jna/PToPtr
  (->ptr-backing-store [item]
    ;;There should be a check here so that only devices that support
    ;;pointer offset allow this call.  Other calls should be via
    ;;the base-ptr protocol
    (-> (base-ptr item)
        Pointer/nativeValue
        (+ (long (byte-offset item)))
        (Pointer.)))
  dtype-base/PDatatype
  (get-datatype [item] (dl-datatype->datatype (.dtype item)))
  mp/PElementCount
  (element-count [item] (apply * (mp/get-shape item)))
  dtype-base/PAccess
  (set-value! [ptr ^long offset value]
    (check-cpu-tensor ptr)
    (dtype-base/set-value!
     (dtype-jna/->typed-pointer ptr)
     offset value))
  (set-constant! [ptr offset value elem-count]
    (check-cpu-tensor ptr)
    (dtype-base/set-constant! (dtype-jna/->typed-pointer ptr) offset value elem-count))
  (get-value [ptr ^long offset]
    (check-cpu-tensor ptr)
    (dtype-base/get-value (dtype-jna/->typed-pointer ptr) offset))
  resource/PResource
  (release-resource [ary]
    (check-call (TVMArrayFree ary)))

  ;;Do jna buffer to take advantage of faster memcpy, memset, and
  ;;other things jna datatype bindings provide.
  dtype-base/PContainerType
  (container-type [ptr] :jna-buffer)

  dtype-base/PCopyRawData
  (copy-raw->item! [raw-data ary-target target-offset options]
    (check-cpu-tensor raw-data)
    (dtype-base/copy-raw->item! (dtype-jna/->typed-pointer raw-data) ary-target
                                target-offset options))

  primitive/PToBuffer
  (->buffer-backing-store [src]
    (check-cpu-tensor src)
    (primitive/->buffer-backing-store (dtype-jna/->typed-pointer src)))

  primitive/PToArray
  (->array [src] nil)
  (->array-copy [src]
    (check-cpu-tensor src)
    (primitive/->array-copy (dtype-jna/->typed-pointer src)))


  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (dtype-base/->vector (dtype-jna/->TypedPointer
                                       (.shape m)
                                       (* (.ndim m) Long/BYTES)
                                       :int64)))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))

  bindings-proto/PTVMDeviceId
  (device-id [item]
    (.device_id (.ctx item)))

  bindings-proto/PTVMDeviceType
  (device-type [item]
    (device-type-int->device-type
     (.device_type (.ctx item))))

  bindings-proto/PByteOffset
  (byte-offset [item] (.byte_offset item))
  (base-ptr [item] (.data item)))



(make-tvm-jna-fn TVMArrayAlloc
                 "Allocate a new tvm array"
                 Integer
                 [shape ->long-ptr]
                 [n-dim int]
                 [dtype_code int]
                 [dtype_bits int]
                 [dtype_lanes int]
                 [device_type device-type->int]
                 [device_id device-id->int]
                 [retval ptr-ptr])


(defn allocate-device-array
  ^DLPack$DLTensor [shape datatype device-type ^long device-id]
  (let [n-dims (dtype/ecount shape)
        ^DLPack$DLDataType dl-dtype (datatype->dl-datatype datatype)
        retval-ptr (PointerByReference.)]
    (check-call
     (TVMArrayAlloc shape n-dims
                    (.code dl-dtype) (.bits dl-dtype) (.lanes dl-dtype)
                    device-type device-id
                    retval-ptr))
    (-> (DLPack$DLTensor. (.getValue retval-ptr))
        resource/track)))


(defn ensure-tensor
  [item]
  (let [item (->tvm item)]
    (when-not (instance? DLPack$DLTensor item)
      (throw (ex-info "Item not a tensor"
                      {:item-type (type item)})))
    item))


(defn to-size-t
  [item]
  (case Native/SIZE_T_SIZE
    8 (long item)
    4 (int item)))


(make-tvm-jna-fn TVMArrayCopyFromBytes
                 "Copy bytes into an array"
                 Integer
                 [dest-tensor ensure-tensor]
                 [src checknil]
                 [n-bytes to-size-t])


(defn copy-to-array!
  [src dest-tensor ^long n-bytes]
  (check-call (TVMArrayCopyFromBytes dest-tensor src n-bytes)))


(make-tvm-jna-fn TVMArrayCopyToBytes
                 "Copy tensor data to bytes"
                 Integer
                 [src-tensor ensure-tensor]
                 [dest checknil]
                 [n-bytes to-size-t])


(defn copy-from-array!
  [src-tensor ^Pointer dest ^long n-bytes]
  (check-call (TVMArrayCopyToBytes src-tensor dest n-bytes)))


(make-tvm-jna-fn TVMArrayCopyFromTo
                 "Copy data from an array to an array"
                 Integer
                 [src ensure-tensor]
                 [dst ensure-tensor]
                 [stream stream/ensure-stream->ptr])


(defn copy-array-to-array!
  [src dst stream]
  (let [stream (if stream
                 stream
                 (stream/->StreamHandle 0 0 (Pointer. 0)))]
    (check-call (TVMArrayCopyFromTo src dst stream))))


(defn pointer->tvm-ary
  "Not all backends in TVM can offset their pointer types.  For this reason, tvm arrays
  have a byte_offset member that you can use to make an array not start at the pointer's
  base address."
  ^DLPack$DLTensor [ptr device-type device-id
                    datatype shape strides
                    byte-offset]
  (let [shape-ptr (dtype-jna/make-typed-pointer
                   :int64 shape)
        strides-ptr (when strides
                      (dtype-jna/make-typed-pointer
                       :int64 strides))
        datatype (or datatype (dtype/get-datatype ptr))
        ;;Get the real pointer
        address (-> (dtype-jna/->ptr-backing-store ptr)
                    dtype-jna/pointer->address)
        retval (DLPack$DLTensor.)
        ctx (DLPack$DLContext.)]
    (when-not (> address 0)
      (throw (ex-info "Failed to get pointer for buffer."
                      {:original-ptr ptr})))
    (set! (.data retval) (Pointer. address))
    (set! (.device_type ctx) (device-type->int device-type))
    (set! (.device_id ctx) (int device-id))
    (set! (.ctx retval) ctx)
    (set! (.ndim retval) (count shape))
    (set! (.dtype retval) (datatype->dl-datatype datatype))
    (set! (.shape retval) (dtype-jna/->ptr-backing-store shape-ptr))
    (when strides-ptr
      (set! (.strides retval) (dtype-jna/->ptr-backing-store strides-ptr)))
    (set! (.byte_offset retval) (long byte-offset))
    retval))
