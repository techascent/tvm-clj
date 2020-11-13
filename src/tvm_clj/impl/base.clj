(ns tvm-clj.impl.base
  (:require [tech.v3.jna :refer [checknil] :as jna]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.datatype.errors :as errors]
            ;;JNA bindings for dtype datastructures
            [tech.v3.datatype.jna]
            [tvm-clj.impl.library-paths :as jna-lib-paths]
            [tvm-clj.impl.definitions :as definitions]
            [tvm-clj.impl.protocols :refer [->tvm-value ->tvm ->node
                                            device-type device-id byte-offset
                                            base-ptr] :as bindings-proto]
            [clojure.set :as c-set]
            [tech.v3.resource :as resource]
            [clojure.tools.logging :as log]
            [clojure.stacktrace :as st])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [clojure.lang IFn]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor CFunction$TVMPackedCFunc
            CFunction$TVMPackedCFuncFinalizer]))


;;C interface functions.
;; 00000000012906d0 T TVMAPISetLastError
;; 00000000012b78b0 T TVMArrayAlloc
;; 00000000012b72e0 T TVMArrayCopyFromBytes
;; 00000000012b80b0 T TVMArrayCopyFromTo
;; 00000000012b75d0 T TVMArrayCopyToBytes
;; 00000000012b6e60 T TVMArrayFree
;; 00000000012b6fb0 T TVMArrayFromDLPack
;; 00000000012b6e40 T TVMArrayGetTypeIndex
;; 00000000012b7a10 T TVMArrayToDLPack
;; 00000000012938e0 T TVMBackendAllocWorkspace
;; 00000000012934c0 T TVMBackendFreeWorkspace
;; 0000000001292560 T TVMBackendGetFuncFromEnv
;; 00000000012c18f0 T TVMBackendParallelBarrier
;; 00000000012c1970 T TVMBackendParallelLaunch
;; 00000000012c0e10 T TVMBackendRegisterSystemLibSymbol
;; 00000000012907a0 T TVMBackendRunOnce
;; 0000000001295a20 T TVMCbArgToReturn
;; 00000000012958e0 T TVMCFuncSetReturn
;; 0000000001292dc0 T TVMDeviceAllocDataSpace
;; 0000000001292fc0 T TVMDeviceCopyDataFromTo
;; 0000000001292ed0 T TVMDeviceFreeDataSpace
;; 00000000012b6eb0 T TVMDLManagedTensorCallDeleter
;; 00000000012945d0 T TVMFuncCall
;; 0000000001292620 T TVMFuncCreateFromCFunc
;; 00000000012907c0 T TVMFuncFree
;; 00000000012bfe30 T TVMFuncGetGlobal
;; 00000000012bfa80 T TVMFuncListGlobalNames
;; 00000000012c0480 T TVMFuncRegisterGlobal
;; 0000000001290640 T TVMGetLastError
;; 0000000001290790 T TVMModFree
;; 0000000001292410 T TVMModGetFunction
;; 0000000001292310 T TVMModImport
;; 0000000001292000 T TVMModLoadFromFile
;; 00000000012bb220 T TVMObjectDerivedFrom
;; 00000000012ba4b0 T TVMObjectFree
;; 00000000012ba4f0 T TVMObjectGetTypeIndex
;; 00000000012ba490 T TVMObjectRetain
;; 00000000012bc590 T TVMObjectTypeKey2Index
;; 0000000001292a90 T TVMSetStream
;; 0000000001292890 T TVMStreamCreate
;; 0000000001292990 T TVMStreamFree
;; 0000000001292cb0 T TVMStreamStreamSynchronize
;; 0000000001292bb0 T TVMSynchronize


(defmacro make-tvm-jna-fn
  "TVM functions are very regular so the mapping to them can exploit this.
Argpair is of type [symbol type-coersion]."
  [fn-name docstring rettype & argpairs]
  `(jna/def-jna-fn jna-lib-paths/tvm-library-name ~fn-name ~docstring ~rettype ~@argpairs))


(defn keyword->tvm-datatype
  [kwd]
  (definitions/keyword->tvm-datatype kwd))


(defn tvm-datatype->keyword-nothrow
  [tvm-datatype]
  (definitions/tvm-datatype->keyword-nothrow tvm-datatype))


(defn tvm-datatype->keyword
  [tvm-datatype]
  (definitions/tvm-datatype->keyword tvm-datatype))



(defn datatype->dl-type-code
  [datatype]
  (-> (get definitions/datatype->dl-type-code-map datatype)
      keyword->tvm-datatype))

(defn dl-datatype->map
  [^DLPack$DLDataType dtype]
  {:tvm-datatype (tvm-datatype->keyword (.code dtype))
   :bits (.bits dtype)
   :lanes (.lanes dtype)})


(defn dl-datatype->datatype
  [^DLPack$DLDataType dtype]
  (if-let [retval (->> (dl-datatype->map dtype)
                       (get definitions/dl-dtype-map->datatype-map))]
    retval
    (throw (ex-info "Unrecognized datatype"
                    {:dl-datatype->map dtype}))))


(defn datatype->dl-datatype
  [datatype & [dtype-retval]]
  (if-let [retval (get (c-set/map-invert definitions/dl-dtype-map->datatype-map)
                       datatype)]
    (let [^DLPack$DLDataType dtype-retval (or dtype-retval (DLPack$DLDataType.))]
      (set! (.code dtype-retval) (long (keyword->tvm-datatype (:tvm-datatype retval))))
      (set! (.lanes dtype-retval) (long (:lanes retval)))
      (set! (.bits dtype-retval) (long (:bits retval)))
      dtype-retval)
    (throw (ex-info "Failed to find datatype" {:datatype datatype}))))



(defn int-ptr
  ^IntByReference [item]
  (jna/ensure-type IntByReference item))


(defn ptr-ptr
  ^PointerByReference [item]
  (jna/ensure-ptr-ptr item))

(defn long-ptr
  ^LongByReference [item]
  (jna/ensure-type LongByReference item))

(defn ->long-ptr
  [item]
  (if (instance? Pointer item)
    item
    (-> (dtype/make-container :native-buffer :int64 item)
        jna/->ptr-backing-store)))


(defn device-type->int
  [item]
  (let [item (cond (number? item)
                   (int item)
                   (keyword? item)
                   item
                   :else
                   (device-type item))]
    (if (keyword? item)
      (definitions/device-type->device-type-int item)
      (int item))))

(defn device-id->int
  [item]
  (-> (if (satisfies? bindings-proto/PTVMDeviceId item)
        (device-id item)
        item)
      int))


(make-tvm-jna-fn TVMGetLastError
                 "Get last tvm error as byte ptr"
                 Pointer)

(defn get-last-error
  []
  (-> (TVMGetLastError)
      (jna/variable-byte-ptr->string)))


(def ^:dynamic fn-name "")


(defmacro check-call
  [& body]
  `(let [ret# (int (do ~@body))]
     (when-not (= 0 ret#)
       (let [byte-string# (get-last-error)
             fn-name# (if (= "" fn-name)
                        "__unknown_function__"
                        fn-name)]
         (throw (ex-info (format "Error calling TVM fn %s:\n%s" fn-name# byte-string#)
                         {:error-string byte-string#
                          :fn-name fn-name}))))))


(make-tvm-jna-fn TVMFuncListGlobalNames
                 "List the global names"
                 Integer
                 [num-fns int-ptr]
                 [fn-names ptr-ptr])


(defonce global-function-names
  (memoize
   (fn []
     (let [int-data (IntByReference.)
           fn-names (PointerByReference.)]
       (check-call (TVMFuncListGlobalNames int-data fn-names))
       (->> (jna/char-ptr-ptr->string-vec (.getValue int-data)
                                          (.getValue fn-names))
            sort
            vec)))))


(defn find-global-fns
  [^String substr]
  (filter (fn [^String fn-name]
            (.contains (.toLowerCase fn-name)
                       (.toLowerCase substr)))
          (global-function-names)))


(make-tvm-jna-fn TVMFuncGetGlobal
                 "Get a global function ptr"
                 Integer
                 [fn-name str]
                 [fn-ptr ptr-ptr])


(defn name->global-function
  [fn-name]
  (let [retval (PointerByReference.)
        _ (check-call (TVMFuncGetGlobal fn-name retval))
        addr (.getValue retval)]
    (errors/when-not-errorf
     (not= 0 (Pointer/nativeValue addr))
     "Failed to find global function: %s" fn-name)
    addr))


(make-tvm-jna-fn TVMFuncCall
                 "Call a tvm function"
                 Integer
                 [fn-handle checknil]
                 [arg_values jna/as-ptr]
                 [type_codes jna/as-ptr]
                 [num_args int]
                 [ret_val long-ptr]
                 [ret_type_code int-ptr])


(defn arg-list->tvm-args
 [args]
  (let [num-args (count args)
        arg-vals (dtype/make-container :native-buffer :int64 num-args)
        arg-types (dtype/make-container :native-buffer :int32 num-args)]
    (->> args
         (map-indexed
          (fn [idx arg]
            (let [[long-val dtype :as data] (->tvm-value arg)]
              (when-not data
                (throw (Exception. (format "Calling %s - Invalid tvm function argument: (%s) %s"
                                           fn-name (type arg) arg))))
              (dtype/set-value! arg-vals idx long-val)
              (dtype/set-value! arg-types idx
                                (keyword->tvm-datatype dtype)))))
         dorun)
    [arg-vals arg-types num-args]))


(defmulti tvm-value->jvm
  "Attempts to coerce the tvm value into the jvm.  Failures
result in a returned map container a value for the key:
:tvm->jvm-failure

This is in order to ensure that, for instance, deserialization of a node's fields
  allows for a sane recovery mechanism and doesn't lose those field values."
  (fn [long-val val-type-kwd]
    val-type-kwd))

(defmethod tvm-value->jvm :default
  [long-val val-type-kwd]
  (log/warnf "Failed to map value type %s" val-type-kwd)
  [long-val val-type-kwd])

(defmethod tvm-value->jvm :int
  [long-val val-type-kwd]
  long-val)

(defmethod tvm-value->jvm :uint
  [long-val val-type-kwd]
  long-val)

(defmethod tvm-value->jvm :float
  [long-val val-type-kwd]
  (Double/longBitsToDouble long-val))

(defmethod tvm-value->jvm :string
  [long-val val-type-kwd]
  (jna/variable-byte-ptr->string (Pointer. long-val)))

(defmethod tvm-value->jvm :null
  [long-val val-type-kwd]
  nil)


(make-tvm-jna-fn TVMCFuncSetReturn
                 "Set the return value of TVMPackedCFunc.

This function is called by TVMPackedCFunc to set the return value.
When this function is not called, the function returns null by default.

* ret The return value handle, pass by ret in TVMPackedCFunc
* value The value to be returned.
* type_code The type of the value to be returned.
* num_ret Number of return values, for now only 1 is supported."
                 Integer
                 [ret jna/as-ptr]
                 [value jna/ensure-ptr]
                 [type-code jna/ensure-ptr]
                 [num-ret int])


(make-tvm-jna-fn TVMAPISetLastError
                 "Set the last error from a function call."
                 nil
                 [msg str])


(make-tvm-jna-fn TVMFuncCreateFromCFunc
                 "Wrap a TVMPackedCFunc to become a FunctionHandle.

 The resource_handle will be managed by TVM API, until the function is no longer used.

 * `func` The packed C function.
 * `resource_handle` The resource handle from front-end, can be NULL.
 * `fin` The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * `out` the result function handle.

 returns 0 when success, -1 when failure happens"
                 Integer
                 [func identity]
                 [res-hdl jna/as-ptr]
                 [finalizer jna/as-ptr]
                 [out identity])


(defn raw-call-function
  "Call the function but make no attempt to convert the result tuple
  into the jvm."
  [tvm-fn & args]
  (resource/stack-resource-context
   (let [retval (LongByReference.)
         rettype (IntByReference.)
         [tvm-args arg-types n-args] (arg-list->tvm-args args)]
     (check-call
      (TVMFuncCall tvm-fn
                   tvm-args arg-types n-args
                   retval rettype))
     [(.getValue retval)
      (tvm-datatype->keyword-nothrow (.getValue rettype))])))


(defn call-function
  [tvm-fn & args]
  (let [fn-ret-val (apply raw-call-function tvm-fn args)]
    (apply tvm-value->jvm fn-ret-val)))


(make-tvm-jna-fn TVMObjectFree
                 "Free a tvm node."
                 Integer
                 [handle checknil])


(make-tvm-jna-fn TVMFuncFree
                 "Free a tvm function"
                 Integer
                 [handl (jna/as-ptr)])


(defmacro ^:private impl-tvm-ifn
  []
  `(deftype ~'TVMFunction [~'handle ~'gc-obj]
     bindings-proto/PJVMTypeToTVMValue
     (->tvm-value [this#] [(Pointer/nativeValue ~'handle) :func-handle])
     bindings-proto/PToTVM
     (->tvm [this#] this#)
     jna/PToPtr
     (is-jna-ptr-convertible? [this#] true)
     (->ptr-backing-store [this#] ~'handle)
     IFn
     ~@(->> (range 16)
         (map (fn [idx]
                (let [argsyms (->> (range idx)
                                   (map (fn [arg-idx]
                                          (symbol (str "arg-" arg-idx)))))]
                  `(invoke ~(vec (concat ['this]
                                         argsyms))
                           (call-function ~'handle ~@argsyms))))))
     (applyTo [this# argseq#]
       (apply call-function ~'handle argseq#))
     (run [this#]
       (TVMObjectFree ~'handle))
     java.lang.AutoCloseable
     (close [this#]
       (TVMObjectFree ~'handle))))


(impl-tvm-ifn)


(defn clj-fn->tvm-fn
  "Given a clojure IFn implementation, return a tvm packed function that TVM can call
  or you can call using call-function.."
  [clj-fn]
  (let [iface (reify CFunction$TVMPackedCFunc
                (invoke [this args typecodes num-args ret-val-handle _resource-handle]
                  (let [num-args (long num-args)
                        clj-args (when-not (== 0 num-args)
                                   (let [arg-buf (native-buffer/wrap-address
                                                  (Pointer/nativeValue args)
                                                  (* num-args Long/BYTES)
                                                  :int64 (dtype-proto/platform-endianness)
                                                  nil)
                                         typecodes (native-buffer/wrap-address
                                                    (Pointer/nativeValue typecodes)
                                                    (* num-args Integer/BYTES)
                                                    :int32 (dtype-proto/platform-endianness)
                                                    nil)]
                                     (map (fn [value typecode]
                                            (tvm-value->jvm value (tvm-datatype->keyword-nothrow
                                                                   typecode)))
                                          arg-buf typecodes)))]
                    (try
                      (when-let [retval (apply clj-fn clj-args)]
                        (resource/stack-resource-context
                         (let [[tvm-args arg-types n-args] (arg-list->tvm-args [retval])]
                           ;;n-args is guaranteed to be 1
                           (TVMCFuncSetReturn ret-val-handle tvm-args arg-types 1))))
                      (int 0)
                      (catch Throwable e
                        (let [msg (format "%s:\n%s\n"
                                          (.getMessage e)
                                          (with-out-str
                                            (st/print-stack-trace e 20)))]
                          (TVMAPISetLastError msg))
                        (int -1))))))
        retval-hdl (LongByReference.)
        _ (check-call (TVMFuncCreateFromCFunc iface nil nil retval-hdl))
        ;;The interface instance must stay around as long as the function does.
        retval (TVMFunction. (Pointer. (.getValue retval-hdl)) iface)]
    retval))


(defmethod tvm-value->jvm :func-handle
  [long-val _val-type-kwd]
  (let [long-val (long long-val)
        ptr-data (Pointer. long-val)
        retval (TVMFunction. ptr-data nil)]
    (resource/track retval {:track-type :auto
                            :dispose-fn #(TVMObjectFree ptr-data)})))


(defn global-function
  [fn-name & args]
  (let [fn-data (name->global-function fn-name)]
    (with-bindings {#'fn-name fn-name}
      (apply call-function fn-data args))))


(comment
  (com.sun.jna.NativeLibrary/getInstance
   "/home/chrisn/dev/tech.all/tvm-clj/tvm/build/libtvm.so")
  )
