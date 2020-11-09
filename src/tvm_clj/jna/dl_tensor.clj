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
            [tech.v3.resource :as resource]
            [tvm-clj.bindings.protocols :refer [->tvm
                                                base-ptr
                                                ->tvm-value
                                                byte-offset] :as bindings-proto]
            [tvm-clj.bindings.definitions :refer [device-type-int->device-type]]

            [tech.v3.jna :refer [checknil] :as jna]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.datatype.casting :as casting]
            [tech.v3.datatype.native-buffer :as native-buffer]
            ;;JNA bindings for native buffers
            [tech.v3.datatype.jna]
            [tech.v3.tensor.dimensions.analytics :as dims-analytics]
            [tech.v3.tensor.pprint :as dtt-pprint]
            [tech.v3.tensor :as dtt]
            [clojure.tools.logging :as log])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]
           [java.io Writer]
           [tech.v3.datatype NDBuffer]
           [tech.v3.datatype.native_buffer NativeBuffer]))


(set! *warn-on-reflection* true)


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
                 [item jna/ensure-ptr])


(defn- item-device-type
  [^DLPack$DLTensor item]
  (-> item
      (.ctx)
      (.device_type)
      device-type-int->device-type))


(defn is-cpu-tensor?
  [^DLPack$DLTensor item]
  (= :cpu (item-device-type item)))


(defn check-cpu-tensor
  [item]
  (when-not (is-cpu-tensor? item)
    (throw (ex-info "Item is not a cpu tensor"
                    {:item-device (item-device-type item)})))
    item)


(declare allocate-device-array)


(defn ->native-buffer
  ^NativeBuffer [^DLPack$DLTensor dl-tensor]
  (let [tens-dtype (dtype/elemwise-datatype dl-tensor)]
    (native-buffer/wrap-address (Pointer/nativeValue
                                 (jna/->ptr-backing-store dl-tensor))
                                (* (dtype/ecount dl-tensor)
                                   (casting/numeric-byte-width tens-dtype))
                                tens-dtype (dtype-proto/platform-endianness)
                                dl-tensor)))


(declare pointer->tvm-ary)


(defn- long-array-ptr->jvm
  ^NativeBuffer [^Pointer ary ^long n-elems src]
  (let [nvalue (Pointer/nativeValue ary)]
    (native-buffer/wrap-address
     nvalue
     (* n-elems Long/BYTES)
     :int64
     (dtype-proto/platform-endianness)
     src)))


(defn dl-tensor-strides
  [^DLPack$DLTensor item]
  (if-let [strides (.strides item)]
    (long-array-ptr->jvm strides (.ndim item) item)
    (-> (dtype/shape item)
        (dims-analytics/shape-ary->strides))))


(extend-type DLPack$DLTensor
  bindings-proto/PToTVM
  (->tvm [item] item)
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [item]
    (.write item)
    [(-> (.getPointer item)
         Pointer/nativeValue)
     :array-handle])

  jna/PToPtr
  (is-jna-ptr-convertible? [item] true)
  (->ptr-backing-store [item]
    ;;There should be a check here so that only devices that support
    ;;pointer offset allow this call.  Other calls should be via
    ;;the base-ptr protocol
    (-> (base-ptr item)
        Pointer/nativeValue
        (+ (long (byte-offset item)))
        (Pointer.)))
  dtype-proto/PToNativeBuffer
  (convertible-to-native-buffer? [item] true)
  (->native-buffer [item] (->native-buffer item))
  dtype-proto/PElemwiseDatatype
  (elemwise-datatype [item] (dl-datatype->datatype (.dtype item)))
  dtype-proto/PShape
  (shape [item]
    (vec (long-array-ptr->jvm (.shape item)
                              (.ndim item)
                              item)))
  dtype-proto/PECount
  (ecount [item] (apply * (dtype/shape item)))
  dtype-proto/PClone
  (clone [item]
    (dtype/copy! item
                 (allocate-device-array (dtype/shape item)
                                        (dtype/get-datatype item)
                                        (bindings-proto/device-type item)
                                        (bindings-proto/device-id item))))
  dtype-proto/PToNDBufferDesc
  (convertible-to-nd-buffer-desc? [item] (is-cpu-tensor? item))
  (->nd-buffer-descriptor [item]
    (check-cpu-tensor item)
    ;;Get a typed buffer to get the pointer offsetting correct.
    (let [typed-buf (->native-buffer item)
          ;;link the ptr to the item with the gc system.
          item-dtype (dtype/get-datatype item)]
      {:ptr (.address typed-buf)
       :elemwise-datatype item-dtype
       :datatype :tvm-tensor
       :shape (dtype/shape item)
       :strides (->> (dl-tensor-strides item)
                     (mapv (partial * (casting/numeric-byte-width item-dtype))))
       :source item}))

  dtype-proto/PToTensor
  (as-tensor [item]
    (check-cpu-tensor item)
    (-> (dtype-proto/->nd-buffer-descriptor item)
        (dtt/nd-buffer-descriptor->tensor)))

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


(defn dl-tensor->string
  ^String [dl-tens]
  (format "#tvm.tensor<%s,%s>%s\n%s"
          (name (item-device-type dl-tens))
          (name (dtype/get-datatype dl-tens))
          (dtype/shape dl-tens)
          (if (is-cpu-tensor? dl-tens)
            (dtt-pprint/base-tensor->string (dtt/ensure-tensor dl-tens))
            "{device-data}")))


(defmethod print-method DLPack$DLTensor
  [tens w]
  (.write ^Writer w (dl-tensor->string tens)))


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
  (^DLPack$DLTensor [shape datatype device-type device-id
                     {:keys [resource-type log-level]}]
   (let [n-dims (dtype/ecount shape)
         ^DLPack$DLDataType dl-dtype (datatype->dl-datatype datatype)
         retval-ptr (PointerByReference.)]
     (check-call
      (TVMArrayAlloc shape n-dims
                     (.code dl-dtype) (.bits dl-dtype) (.lanes dl-dtype)
                     device-type device-id
                     retval-ptr))
     (let [retval (DLPack$DLTensor. (.getValue retval-ptr))
           address (Pointer/nativeValue (.data retval))]
       (when log-level
         (log/logf log-level "allocating:  %s : 0x%016X" shape address))
       ;;We allow the gc to help us clean up these things.
       (resource/track retval
                       {:dispose-fn #(do
                                       (when log-level
                                         (log/logf log-level "freeing:  %s : 0x%016X" shape address))
                                       (TVMArrayFree (.getValue retval-ptr)))
                        :track-type (or resource-type :auto)}))))
  (^DLPack$DLTensor [shape datatype device-type ^long device-id]
   (allocate-device-array shape datatype device-type device-id nil)))


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


(defn- untracked-long-data
  ^Pointer [data-ary]
  (-> (dtype/copy! data-ary
                   (-> (native-buffer/malloc
                        (* (dtype/ecount data-ary) Long/BYTES)
                        {:resource-type nil})
                       (native-buffer/set-native-datatype :int64)))
      (jna/->ptr-backing-store)))


(defn pointer->tvm-ary
  "Not all backends in TVM can offset their pointer types.  For this reason, tvm arrays
  have a byte_offset member that you can use to make an array not start at the
  pointer's base address.  If provided this new object will keep the gc-root alive
  in the eyes of the gc at the cost of an extra gc cycle in order to clean up
  gc root.  Ideally caller maintains a reference to gc-root, not this method."
  ^DLPack$DLTensor [ptr device-type device-id
                    datatype shape strides
                    byte-offset {:keys [gc-root resource-type log-level]}]
  ;;Allocate child pointers untracked.  We have to manually track them because on the
  ;;actual dl-tensor object, they are stored in separate structures that only share the
  ;;long address.  Thus the GC, after this method, believes that the shape-ptr and
  ;;strides-ptr objects are free to be cleaned up.
  (let [shape-ptr (untracked-long-data shape)
        strides-ptr (when strides (untracked-long-data strides))
        datatype (or datatype (dtype/elemwise-datatype ptr))

        ;;Get the real pointer.  In the case of something like an opencv image it
        ;;really has two jna ptrs, one to the image object and one to the data buffer.
        ;;Because of this we first convert to a native-buffer which implies a pointer to
        ;;the data buffer and then convert to a jna ptr.
        address (long (if (number? ptr)
                        ptr
                        (-> (dtype-proto/->native-buffer ptr)
                            (jna/->ptr-backing-store)
                            (Pointer/nativeValue))))
        retval (DLPack$DLTensor.)
        ctx (DLPack$DLContext.)
        gc-root-options {:gc-root gc-root}]
    (when log-level
      (log/logf log-level "wrapping:  %s : 0x%016X" shape address))
    (when-not (> address 0)
      (throw (ex-info "Failed to get pointer for buffer."
                      {:original-ptr ptr})))
    (set! (.data retval) (Pointer. address))
    (set! (.device_type ctx) (device-type->int device-type))
    (set! (.device_id ctx) (int device-id))
    (set! (.ctx retval) ctx)
    (set! (.ndim retval) (count shape))
    (set! (.dtype retval) (datatype->dl-datatype datatype))
    (set! (.shape retval) (jna/->ptr-backing-store shape-ptr))
    (when strides-ptr
      (set! (.strides retval) (jna/->ptr-backing-store strides-ptr)))
    (set! (.byte_offset retval) (long byte-offset))

    ;;Attach any free calls to the dl-tensor object itself.  Not to its data members.
    (resource/track
     retval
     {:dispose-fn
      #(do
         ;;This is important to establish a valid chain of scopes for the gc system
         ;;between whatever is providing the ptr data *and* the derived tensor
         (when log-level
           (log/logf log-level "unwrapping:  %s : 0x%016X" shape address))
         ;;Call into the gc root object with a general protocol so it is rooted in this
         ;;closure.  Then throw it away.
         (get gc-root-options :gc-root)
         (native-buffer/free (Pointer/nativeValue shape-ptr))
         (when strides-ptr
           (native-buffer/free (Pointer/nativeValue strides-ptr))))
      :track-type (or resource-type :auto)})))


(defn buffer-desc->dl-tensor
  ([{:keys [ptr elemwise-datatype shape strides] :as descriptor}
    device-type device-id]
   (let [canonical-strides (->> (dims-analytics/shape-ary->strides shape)
                                (mapv (partial * (casting/numeric-byte-width
                                                  elemwise-datatype))))
         strides (if-not (= canonical-strides strides)
                   (mapv #(/ % (casting/numeric-byte-width elemwise-datatype))
                         strides)
                   nil)]
     (pointer->tvm-ary ptr device-type device-id elemwise-datatype shape strides 0 descriptor)))
  ([descriptor]
   (buffer-desc->dl-tensor descriptor :cpu 0)))


(extend-type NDBuffer
  bindings-proto/PToTVM
  (->tvm [item]
    (if-let [buf-desc (dtype-proto/->nd-buffer-descriptor item)]
      (buffer-desc->dl-tensor buf-desc
                              (bindings-proto/device-type item)
                              (bindings-proto/device-id item))
      (throw (Exception. "Tensor is not native-buffer-backed"))))
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [item]
    (if-let [tvm-data (bindings-proto/->tvm item)]
      (bindings-proto/->tvm-value tvm-data)
      (throw (Exception. (format "Invalid tvm function argument: %s" item))))))
