(ns tvm-clj.tvm-jna
  (:require [clojure.set :as c-set]
            [tech.datatype.jna :as dtype-jna]
            [tech.datatype :as dtype]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.protocols :as mp]
            [tech.resource :as resource])
  (:import [com.sun.jna Native NativeLibrary Pointer Function]
           [com.sun.jna.ptr PointerByReference IntByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)



(defprotocol PJVMTypeToTVMValue
  "Convert something to a [long tvm-value-type] pair"
  (->tvm-value [jvm-type]))


(defn- string->ptr
  ^Pointer [^String data]
  (let [str-bytes (.getBytes data "ASCII")
        num-bytes (+ (alength str-bytes) 1)
        typed-data (dtype-jna/make-typed-pointer :int8 num-bytes)]
    (dtype/set-constant! typed-data 0 0 (dtype/ecount typed-data))
    (dtype/copy! str-bytes typed-data)
    (dtype-jna/->ptr-backing-store typed-data)))


(declare tvm-array tvm-map)

(extend-protocol PJVMTypeToTVMValue
  Double
  (->tvm-value [value] [(Double/doubleToLongBits (double value)) :float])
  Float
  (->tvm-value [value] [(Double/doubleToLongBits (double value)) :float])
  Byte
  (->tvm-value [value] [(long value) :int])
  Short
  (->tvm-value [value] [(long value) :int])
  Integer
  (->tvm-value [value] [(long value) :int])
  Long
  (->tvm-value [value] [(long value) :int])
  Boolean
  (->tvm-value [value] [(if value
                             (long 1)
                             (long 0)) :int])
  String
  (->tvm-value [value] [(Pointer/nativeValue (string->ptr value))
                        :string])
  Object
  (->tvm-value [value]
    (cond
      (sequential? value)
      (->tvm-value (apply tvm-array value))
      (map? value)
      (->tvm-value (apply tvm-map (->> (seq value)
                                               (apply concat))))
      (nil? value)
      [(long 0) :null]))

  nil
  (->tvm-value [value]
    [(long 0) :null]))


(defprotocol PToTVM
  "Convert something to some level of tvm type."
  (->tvm [item]))


(defprotocol PConvertToNode
  (->node [item]))



(def tvm-datatype->keyword-map
  {0 :int
   1 :uint
   2 :float
   3 :handle
   4 :null
   5 :tvm-type
   6 :tvm-context
   7 :array-handle
   8 :node-handle
   9 :module-handle
   10 :func-handle
   11 :string
   12 :bytes})


(def datatype->dl-type-code-map
  {:uint8 :uint
   :uint16 :uint
   :uint32 :uint
   :uint64 :uint
   :int8 :int
   :int16 :int
   :int32 :int
   :int64 :int
   :float32 :float
   :float64 :float})

(defn keyword->tvm-datatype
  [kwd]
  (if-let [retval (get (c-set/map-invert tvm-datatype->keyword-map) kwd)]
    retval
    (throw (ex-info "Failed to get tvm-datatype from kwd"
                    {:kwd kwd}))))

(defn tvm-datatype->keyword
  [tvm-datatype]
  (if-let [retval (get tvm-datatype->keyword-map tvm-datatype)]
    retval
    (throw (ex-info "Failed to find keyword for tvm datatype"
                    {:tvm-datatype tvm-datatype}))))

(defn datatype->dl-type-code
  [datatype]
  (-> (get datatype->dl-type-code-map datatype)
      keyword->tvm-datatype))

(defn dl-datatype->map
  [^DLPack$DLDataType dtype]
  {:tvm-datatype (tvm-datatype->keyword (.code dtype))
   :bits (.bits dtype)
   :lanes (.lanes dtype)})


(def dl-dtype-map->datatype-map
  {{:tvm-datatype :float
    :bits 32
    :lanes 1} :float32
   {:tvm-datatype :float
     :bits 64
    :lanes 1} :float64

   {:tvm-datatype :int
    :bits 8
    :lanes 1} :int8
   {:tvm-datatype :int
    :bits 16
    :lanes 1} :int16
   {:tvm-datatype :int
    :bits 32
    :lanes 1} :int32
   {:tvm-datatype :int
    :bits 64
    :lanes 1} :int64

   {:tvm-datatype :uint
    :bits 8
    :lanes 1} :uint8
   {:tvm-datatype :uint
    :bits 16
    :lanes 1} :uint16
   {:tvm-datatype :uint
    :bits 32
    :lanes 1} :uint32
   {:tvm-datatype :uint
    :bits 64
    :lanes 1} :uint64})


(defn dl-datatype->datatype
  [^DLPack$DLDataType dtype]
  (if-let [retval (->> (dl-datatype->map dtype)
                       (get dl-dtype-map->datatype-map))]
    retval
    (throw (ex-info "Unrecognized datatype"
                    {:dl-datatype->map dtype}))))


(defn datatype->dl-datatype
  [datatype & [dtype-retval]]
  (if-let [retval (get (c-set/map-invert dl-dtype-map->datatype-map) datatype)]
    (let [^DLPack$DLDataType dtype-retval (or dtype-retval (DLPack$DLDataType.))]
      (set! (.code dtype-retval) (long (keyword->tvm-datatype (:tvm-datatype retval))))
      (set! (.lanes dtype-retval) (long (:lanes retval)))
      (set! (.bits dtype-retval) (long (:bits retval)))
      dtype-retval)
    (throw (ex-info "Failed to find datatype" {:datatype datatype}))))


(def node-type-name->keyword-map
  {;;Container
   "Array" :array
   "Map" :map
   "Range" :range
   "LoweredFunc" :lowered-function
   ;;Expression
   "Expr" :expression
   "Variable" :variable
   "Reduce" :reduce
   "FloatImm" :float-imm
   "IntImm" :int-imm
   "UIntImm" :uint-imm
   "StringImm" :string-imm
   "Cast" :cast
   "Add" :+
   "Sub" :-
   "Mul" :*
   "Div" :/
   "Min" :min
   "Max" :max
   "EQ" :=
   "NE" :!=
   "LT" :<
   "LE" :<=
   "GT" :>
   "GE" :>=
   "And" :and
   "Not" :!
   "Select" :select
   "Load" :load
   "Ramp" :ramp
   "Broadcast" :broadcast
   "Shuffle" :shuffle
   "Call" :call
   "Let" :let
   ;;Schedule
   "Buffer" :buffer
   "Split" :split
   "Fuse" :fuse
   "IterVar" :iteration-variable
   "Schedule" :schedule
   "Stage" :stage
   ;;Tensor
   "Tensor" :tensor
   "PlaceholderOp" :placeholder-operation
   "ComputeOp" :compute-operation
   "ScanOp" :scan-operation
   "ExternOp" :external-operation
   ;;Statement
   "LetStmt" :let
   "AssertStmt" :assert
   "ProducerConsumer" :producer-consumer
   "For" :for
   "Store" :store
   "Provide" :provide
   "Allocate" :allocate
   "AttrStmt" :attribute
   "Free" :free
   "Realize" :realize
   "Block" :block
   "IfThenElse" :if-then-else
   "Evaluate" :evaluate
   "Prefetch" :prefetch
   ;;build-module
   "BuildConfig" :build-config
   ;;arith.py
   "IntervalSet" :interval-set
   "StrideSet" :stride-set
   "ModularSet" :modular-set
   })


(def expression-set
  #{:expression
    :variable
    :reduce
    :float-imm
    :int-imm
    :uint-imm
    :string-imm
    :cast
    :+
    :-
    :*
    :/
    :min
    :max
    :=
    :!=
    :<
    :<=
    :>
    :>=
    :and
    :!
    :select
    :load
    :ramp
    :broadcast
    :shuffle
    :call
    :let})


(defn is-expression-node?
  [node]
  (expression-set (:tvm-type-kwd node)))


(def kwd->device-type-map
  {:cpu 1
   :cpu-pinned 3
   :cuda 2
   :ext-dev 12
   :gpu 2
   :llvm 1
   :metal 8
   :opencl 4
   :rocm 10
   :stackvm 1
   :vpi 9
   :vulkan 7})

(def device-type->kwd-map (c-set/map-invert kwd->device-type-map))


(defn device-type->device-type-int
  ^long [device-type]
  (if-let [dev-enum (kwd->device-type-map device-type)]
    dev-enum
    (throw (ex-info "Failed to find device type enum"
                    {:device-type device-type}))))


(defn device-type-int->device-type
  [^long device-type]
  (if-let [retval (device-type->kwd-map device-type)]
    retval
    (throw (ex-info "Failed to find keyword for device type"
                    {:device-type device-type}))))


(def load-library
  (memoize
   (fn [libname]
     (NativeLibrary/getInstance libname))))


(def do-find-function
  (memoize
   (fn [fn-name libname]
     (.getFunction ^NativeLibrary (load-library libname) fn-name))))

(defn find-function
  ^Function [fn-name libname]
  (do-find-function fn-name libname))



(def ^:dynamic *tvm-library-name* "tvm")


(def ^:dynamic fn-name "")



(defn unsafe-read-byte
  [^Pointer byte-ary ^long idx]
  (.get (.getByteBuffer byte-ary idx 1) 0))


(defn variable-byte-ptr->string
  [^long ptr-addr]
  (if (= 0 ptr-addr)
    ""
    (let [byte-ary (Pointer. ptr-addr)]
      (String. ^"[B"
               (into-array Byte/TYPE
                           (take-while #(not= % 0)
                                       (map #(unsafe-read-byte
                                              byte-ary %)
                                            (range))))))))

(defn- to-typed-fn
  ^Function [item] item)


(defmacro make-tvm-jna-fn
  "TVM functions are very regular so the mapping to them can exploit this.
Argpair is of type [symbol type-coersion]."
  [fn-name docstring rettype & argpairs]
  `(defn ~fn-name
     ~docstring
     ~(mapv first argpairs)
     (let [~'tvm-fn (find-function ~(str fn-name) *tvm-library-name*)
           ~'fn-args (object-array ~(mapv (fn [[arg-symbol arg-coersion]]
                                            `(~arg-coersion ~arg-symbol))
                                          argpairs))]
       ~(if rettype
          `(.invoke (to-typed-fn ~'tvm-fn) ~rettype ~'fn-args)
          `(.invoke (to-typed-fn ~'tvm-fn) ~'fn-args)))))


(make-tvm-jna-fn TVMGetLastError
                 "Get last tvm error as byte ptr"
                 Pointer)

(defn get-last-error
  []
  (-> (TVMGetLastError)
      (Pointer/nativeValue)
      variable-byte-ptr->string))


(defmacro check-call
  [& body]
  `(let [ret# (int (do ~@body))]
     (when-not (= 0 ret#)
       (let [byte-string# (get-last-error)]
         (throw (ex-info (format "Error during TVM call: %s" byte-string#)
                         {:error-string byte-string#
                          :fn-name fn-name}))))))


(defn int-ptr
  ^IntByReference [item]
  (when-not (instance? IntByReference item)
    (throw (ex-info "Item is not an int-ptr"
                    {:item item})))
  item)


(defn ptr-ptr
  ^PointerByReference [item]
  (when-not (instance? PointerByReference item)
    (throw (ex-info "Item is not a ptr-ptr"
                    {:item item})))
  item)


(make-tvm-jna-fn TVMFuncListGlobalNames
                 "List the global names"
                 Integer
                 [num-fns int-ptr]
                 [fn-names ptr-ptr])


(def global-function-names
  (memoize
   (fn []
     (let [int-data (IntByReference.)
           fn-names (PointerByReference.)
           _ (check-call (TVMFuncListGlobalNames int-data fn-names))
           base-address (Pointer/nativeValue (.getValue fn-names))]
       (->> (range (.getValue int-data))
            (map (fn [name-idx]
                   (let [new-ptr (-> (+ (* (long name-idx) Native/POINTER_SIZE)
                                        base-address)
                                     (Pointer.))
                         char-ptr (case Native/POINTER_SIZE
                                    8 (.getLong new-ptr 0)
                                    4 (.getInt new-ptr 0))]
                     (variable-byte-ptr->string char-ptr))))
            sort
            vec)))))


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
    (when (= 0 (Pointer/nativeValue addr))
      (throw (ex-info "Failed to find global function"
                      {:fn-name fn-name})))
    addr))


(defn checknil
  ^Pointer [value]
  (if (instance? Pointer value)
    (checknil (Pointer/nativeValue value))
    (if (= 0 (long value))
      (throw (ex-info "Pointer value is nil"
                      {}))
      (Pointer. value))))


(defn- tvm-value->long
  ^long [^Pointer value]
  (checknil value)
  (let [bb (.getByteBuffer value 0 8)
        lb (.asLongBuffer bb)]
    (.get lb 0)))


(defn- ensure-type
  [item-cls item]
  (when-not (instance? item-cls item)
    (throw (ex-info "Item is not desired type"
                    {:item-cls item-cls
                     :item item})))
  item)


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


(defn- check-cpu-tensor
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


;;We have to wrap the dl-tensor to take care of the situation
;;where, for instance, it does not own the memory.
(extend-type DLPack$DLTensor
  PToTVM
  (->tvm [item] item)
  PJVMTypeToTVMValue
  (->tvm-value [item]
    (.write item)
    [(-> (.getPointer item)
         Pointer/nativeValue)
     :array-handle])
  dtype-jna/PToPtr
  (->ptr-backing-store [item]
    (.data item))
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
                         :shape shape}))))))


(defn- ->long-ptr
  [item]
  (if (instance? Pointer item)
    item
    (-> (dtype-jna/make-typed-pointer :int64 item)
        dtype-jna/->ptr-backing-store)))


(make-tvm-jna-fn TVMArrayAlloc
                 "Allocate a new tvm array"
                 Integer
                 [shape ->long-ptr]
                 [n-dim int]
                 [dtype_code int]
                 [dtype_bits int]
                 [dtype_lanes int]
                 [device_type int]
                 [device_id int]
                 [retval ptr-ptr])


(defn allocate-device-array
  ^DLPack$DLTensor [shape datatype device-type ^long device-id]
  (let [n-dims (dtype/ecount shape)
        ^DLPack$DLDataType dl-dtype (datatype->dl-datatype datatype)
        device-type-int (int (if (number? device-type)
                               device-type
                               (device-type->device-type-int device-type)))
        retval-ptr (PointerByReference.)]
    (check-call
     (TVMArrayAlloc shape n-dims
                    (.code dl-dtype) (.bits dl-dtype) (.lanes dl-dtype)
                    device-type-int device-id
                    retval-ptr))
    retval-ptr))


(defrecord StreamHandle [^long device ^long dev-id ^long tvm-hdl]
  PToTVM
  (->tvm [item] item)
  PJVMTypeToTVMValue
  (->tvm-value [item] (throw (ex-info "Unsupported" {}))))


(defrecord ModuleHandle [^long tvm-hdl]
  PToTVM
  (->tvm [item] item)
  PJVMTypeToTVMValue
  (->tvm-value [item] [tvm-hdl :module-handle]))



;; (def node-type-name->index
;;   (memoize get-type-key-for-name))

;; ;;May return nil
;; (def node-type-index->keyword
;;   (memoize
;;    (fn [type-index]
;;      (->> node-type-name->keyword-map
;;           (map (fn [[type-name keyword]]
;;                  (when (= type-index (node-type-name->index type-name))
;;                    keyword)))
;;           (remove nil?)
;;           first))))
