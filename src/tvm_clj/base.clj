(ns tvm-clj.base
  "Base types for the tvm-clj system.  This avoids the issue where a recompilation leaves
  protocols in difficult-to-understand states or redefines records.  Nothing should be defined
  in this file but types; this allows repl recompilation to succeed predictably"
  (:require [potemkin :as p]
            [tech.datatype.core :as dtype]
            [tech.resource :as resource]
            [tech.datatype.javacpp :as jcpp-dtype]
            [tech.datatype.java-unsigned :as unsigned])
  (:import [tvm_clj.tvm runtime runtime$TVMFunctionHandle runtime$TVMValue
            runtime$NodeHandle runtime$TVMModuleHandle
            runtime$DLTensor runtime$TVMStreamHandle
            runtime$DLTensor runtime$DLContext]
           [org.bytedeco.javacpp Pointer LongPointer]
           [java.lang.reflect Field]))


(defprotocol PJVMTypeToTVMValue
  "Convert something to a [long tvm-value-type] pair"
  (->tvm-value [jvm-type]))


(defprotocol PToTVM
  "Convert something to some level of tvm type."
  (->tvm [item]))


(extend-protocol PToTVM
  runtime$TVMFunctionHandle
  (->tvm [item] item)
  runtime$TVMValue
  (->tvm [item] item)
  runtime$NodeHandle
  (->tvm [item] item)
  runtime$TVMModuleHandle
  (->tvm [item] item)
  runtime$DLTensor
  (->tvm [item] item)
  runtime$TVMStreamHandle
  (->tvm [item] item))


(defprotocol PConvertToNode
  (->node [item]))


(defrecord ArrayHandle [^runtime$DLTensor tvm-jcpp-handle]
  PToTVM
  (->tvm [_] tvm-jcpp-handle))

(defrecord StreamHandle [^long device ^long dev-id ^runtime$TVMStreamHandle tvm-hdl]
  PToTVM
  (->tvm [_] tvm-hdl))


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


(defn raw-create-tvm-ary
  [^Pointer ptr device-type device-id datatype byte-offset
   ^LongPointer shape-ptr & {:keys [strides-ptr]}]
  (let [tens-data (runtime$DLTensor. 1)
        n-dims (.capacity shape-ptr)
        ctx (.ctx tens-data)
        dtype (.dtype tens-data)]
    (.data tens-data ptr)
    (.ndim tens-data n-dims)
    (.byte_offset tens-data (long byte-offset))
    (.device_type ctx (int device-type))
    (.device_id ctx (int device-id))
    (.code dtype (datatype->dl-type-code datatype))
    (.bits dtype (datatype->dl-bits datatype))
    (.lanes dtype 1)
    (.shape tens-data shape-ptr)
    (if strides-ptr
      (.strides tens-data strides-ptr)
      (.strides tens-data (LongPointer.)))
    tens-data))


(defn pointer->tvm-ary
  "Not all backends in TVM can offset their pointer types.  For this reason, tvm arrays
  have a byte_offset member that you can use to make an array not start at the pointer's
  base address."
  ^ArrayHandle [ptr device-type device-id
                datatype shape strides
                byte-offset]
  (let [^LongPointer shape-ptr (resource/track
                                (jcpp-dtype/make-pointer-of-type
                                 :int64 shape))
        ^LongPointer strides-ptr (when strides
                                   (resource/track
                                    (jcpp-dtype/make-pointer-of-type
                                     :int64 strides)))
        datatype (or datatype (dtype/get-datatype ptr))
        ;;Get the real pointer
        ptr (jcpp-dtype/->ptr-backing-store ptr)]
    (when-not ptr
      (throw (ex-info "Failed to get pointer for buffer."
                      {:original-ptr ptr})))
    (resource/track
     (merge (->ArrayHandle
             (raw-create-tvm-ary ptr device-type device-id datatype
                                 byte-offset shape-ptr
                                 :strides-ptr strides-ptr))
            {:shape shape
             :datatype datatype
             :owns-memory? false}))))


(defn tvm-ary->pointer
  ^Pointer [^ArrayHandle ten-ary ^long elem-count datatype]
  (let [^runtime$DLTensor tensor (.tvm-jcpp-handle ten-ary)
        tens-ptr (.data tensor)
        ptr-dtype (unsigned/datatype->jvm-datatype datatype)
        retval (jcpp-dtype/make-empty-pointer-of-type ptr-dtype)]
    (.set ^Field jcpp-dtype/address-field retval (+ (.address tens-ptr)
                                                    (.byte_offset tensor)))
    (jcpp-dtype/set-pointer-limit-and-capacity retval elem-count)))
