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
            [tvm-clj.bindings.definitions :refer [device-type-int->device-type]]

            [tech.jna :refer [checknil] :as jna]
            [tech.v2.datatype :as dtype]
            [tech.v2.datatype.protocols :as dtype-proto]
            [tech.v2.datatype.jna :as dtype-jna]
            [tech.v2.datatype.casting :as casting]
            [tech.v2.tensor.dimensions :as tens-dims]
            [tech.v2.tensor.pprint :as dtt-pprint]
            [tech.v2.tensor :as dtt]
            [tech.v2.tensor.protocols :as dtt-proto])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]
           [java.io Writer]))


(def ^:dynamic *debug-dl-tensor-lifespan* false)


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


(defn as-typed-buffer
  [^DLPack$DLTensor dl-tensor]
  (dtype-jna/unsafe-ptr->typed-pointer
   (jna/->ptr-backing-store dl-tensor)
   (* (dtype/ecount dl-tensor)
      (casting/numeric-byte-width (dtype/get-datatype dl-tensor)))
   (dtype/get-datatype dl-tensor)))


(declare pointer->tvm-ary)


(defn dl-tensor-strides
  [^DLPack$DLTensor item]
  (if-let [strides (.strides item)]
    (-> (dtype-jna/unsafe-address->typed-pointer
         (Pointer/nativeValue strides)
         (* (.ndim item) Long/BYTES)
         :int64))
    (-> (dtype/shape item)
        (tens-dims/extend-strides))))


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
  dtype-proto/PDatatype
  (get-datatype [item] (dl-datatype->datatype (.dtype item)))
  dtype-proto/PShape
  (shape [item]
    (-> (dtype-jna/unsafe-address->typed-pointer
         (Pointer/nativeValue (.shape item))
         (* (.ndim item) Long/BYTES)
         :int64)
        (dtype/->vector)))
  dtype-proto/PCountable
  (ecount [item] (apply * (dtype/shape item)))
  dtype-proto/PPrototype
  (from-prototype [item datatype shape]
    (allocate-device-array shape datatype
                           (bindings-proto/device-type item)
                           (bindings-proto/device-id item)))
  dtype-proto/PSetConstant
  (set-constant! [item offset value elem-count]
    (check-cpu-tensor item)
    (dtype-proto/set-constant! (as-typed-buffer item) offset value elem-count))
  dtype-proto/PToNioBuffer
  (convertible-to-nio-buffer? [item] (boolean (is-cpu-tensor? item)))
  (->buffer-backing-store [item]
    (dtype-proto/->buffer-backing-store (as-typed-buffer item)))
  dtype-proto/PBuffer
  (sub-buffer [buffer offset length]
    (let [base-ptr (bindings-proto/base-ptr buffer)
          datatype (dtype/get-datatype buffer)
          byte-offset (long (bindings-proto/byte-offset buffer))]
      (pointer->tvm-ary base-ptr
                        (bindings-proto/device-type buffer)
                        (bindings-proto/device-id buffer)
                        datatype
                        [length]
                        nil
                        ;;add the byte offset where the new pointer should start
                        (+ byte-offset
                           (* (long offset)
                              (long (dtype/datatype->byte-size
                                     datatype))))
                        ;;Use the passed in buffer as a gc-root object; obviously the
                        ;;new thing should refer back to the source.
                        buffer)))
  ;;Do jna buffer to take advantage of faster memcpy, memset, and
  ;;other things jna datatype bindings provide.
  dtype-proto/PCopyRawData
  (copy-raw->item! [raw-data ary-target target-offset options]
    (check-cpu-tensor raw-data)
    (dtype-proto/copy-raw->item! (as-typed-buffer raw-data) ary-target
                                 target-offset options))
  dtype-proto/PToArray
  (->sub-array [item])
  (->array-copy [item]
    (check-cpu-tensor item)
    (dtype-proto/->array-copy (as-typed-buffer item)))

  dtype-proto/PToBufferDesc
  (convertible-to-buffer-desc? [item] (is-cpu-tensor? item))
  (->buffer-descriptor [item]
    ;;Get a typed buffer to get the pointer offsetting correct.
    (let [typed-buf (as-typed-buffer item)
          ;;link the ptr to the item with the gc system.
          new-ptr (resource/track
                   (jna/->ptr-backing-store typed-buf)
                   #(bindings-proto/byte-offset item)
                   :gc)
          item-dtype (dtype/get-datatype item)]
      {:ptr new-ptr
       :datatype item-dtype
       :shape (dtype/shape item)
       :strides (->> (dl-tensor-strides item)
                     (mapv (partial * (casting/numeric-byte-width item-dtype))))}))

  dtype-proto/PToWriter
  (convertible-to-writer? [item] (is-cpu-tensor? item))
  (->writer [item options]
    (check-cpu-tensor item)
    (dtype-proto/->writer (as-typed-buffer item) options))

  dtype-proto/PToReader
  (convertible-to-reader? [item] (is-cpu-tensor? item))
  (->reader [item options]
    (check-cpu-tensor item)
    (dtype-proto/->reader (as-typed-buffer item) options))

  dtt-proto/PToTensor
  (tensor-convertible? [item] (is-cpu-tensor? item))
  (convert-to-tensor [item]
    (check-cpu-tensor item)
    (-> (dtype-proto/->buffer-descriptor item)
        (dtt/buffer-descriptor->tensor)))

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
  ^DLPack$DLTensor [shape datatype device-type ^long device-id]
  (println device-type device-id)
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
      (when *debug-dl-tensor-lifespan*
        (println "allocated root tensor of shape" shape ":" address))
      ;;We allow the gc to help us clean up these things.
      (resource/track retval #(do
                                (when *debug-dl-tensor-lifespan*
                                  (println "freeing root tensor of shape"
                                           shape ":" address))
                                (TVMArrayFree (.getValue retval-ptr)))
                      [:stack :gc]))))


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
  have a byte_offset member that you can use to make an array not start at the
  pointer's base address.  If provided this new object will keep the gc-root alive
  in the eyes of the gc."
  ^DLPack$DLTensor [ptr device-type device-id
                    datatype shape strides
                    byte-offset & [gc-root]]
  ;;Allocate child pointers untracked.  We have to manually track them because on the
  ;;actual dl-tensor object, they are stored in separate structures that only share the
  ;;long address.  Thus the GC, after this method, believes that the shape-ptr and
  ;;strides-ptr objects are free to be cleaned up.
  (let [shape-ptr (dtype-jna/make-typed-pointer
                   :int64 shape {:untracked? true})
        strides-ptr (when strides
                      (dtype-jna/make-typed-pointer
                       :int64 strides {:untracked? true}))
        datatype (or datatype (dtype/get-datatype ptr))
        ;;Get the real pointer
        address (-> (jna/->ptr-backing-store ptr)
                    dtype-jna/pointer->address)
        retval (DLPack$DLTensor.)
        ctx (DLPack$DLContext.)
        gc-root-options {:gc-root gc-root}]
    (when *debug-dl-tensor-lifespan*
      (println "deriving from root of shape" shape ":" address))
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
     #(do
        ;;This is important to establish a valid chain of scopes for the gc system
        ;;between whatever is providing the ptr data *and* the derived tensor
        (when *debug-dl-tensor-lifespan*
          (println "freeing derived tensor of shape" shape ":"
                   (-> (jna/->ptr-backing-store ptr)
                       dtype-jna/pointer->address)))
        ;;Call into the gc root object with a general protocol so it is rooted in this
        ;;closure.  Then throw it away.
        (get gc-root-options :gc-root)
        (-> (jna/->ptr-backing-store shape-ptr)
            dtype-jna/unsafe-free-ptr)
        (when strides-ptr
          (-> (jna/->ptr-backing-store strides-ptr)
              dtype-jna/unsafe-free-ptr)))
     [:gc :stack])))


(defn buffer-desc->dl-tensor
  ([{:keys [ptr datatype shape strides] :as descriptor}
    device-type device-id]
   (let [canonical-strides (->> (tens-dims/extend-strides shape)
                                (mapv (partial * (casting/numeric-byte-width
                                                  datatype))))
         strides (if-not (= canonical-strides strides)
                   (mapv #(/ % (casting/numeric-byte-width datatype))
                         strides)
                   nil)]
     (pointer->tvm-ary ptr device-type device-id datatype shape strides 0 descriptor)))
  ([descriptor]
   (buffer-desc->dl-tensor descriptor :cpu 0)))


(extend-type Object
  bindings-proto/PToTVM
  (->tvm [item]
    (if-let [buf-desc (dtype-proto/->buffer-descriptor item)]
      (buffer-desc->dl-tensor buf-desc
                              (bindings-proto/device-type item)
                              (bindings-proto/device-id item))
      (throw (Exception. "Tensor is not native-buffer-backed")))))
