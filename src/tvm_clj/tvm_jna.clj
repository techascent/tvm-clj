(ns tvm-clj.tvm-jna
  (:require [clojure.set :as c-set]
            [tech.datatype.jna :as dtype-jna]
            [tech.datatype :as dtype]
            [tech.datatype.base :as dtype-base]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.protocols :as mp]
            [tech.resource :as resource]
            [tech.jna :as jna]
            [potemkin :as p])
  (:import [com.sun.jna Native NativeLibrary Pointer Function]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)



(defprotocol PJVMTypeToTVMValue
  "Convert something to a [long tvm-value-type] pair"
  (->tvm-value [jvm-type]))


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
  (->tvm-value [value] [(Pointer/nativeValue (jna/string->ptr value))
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


(defn tvm-datatype->keyword-nothrow
  [tvm-datatype]
  (get tvm-datatype->keyword-map tvm-datatype tvm-datatype))

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


(defmacro make-tvm-jna-fn
  "TVM functions are very regular so the mapping to them can exploit this.
Argpair is of type [symbol type-coersion]."
  [fn-name docstring rettype & argpairs]
  `(jna/def-jna-fn *tvm-library-name* ~fn-name ~docstring ~rettype ~@argpairs))


(make-tvm-jna-fn TVMGetLastError
                 "Get last tvm error as byte ptr"
                 Pointer)

(defn get-last-error
  []
  (-> (TVMGetLastError)
      (jna/variable-byte-ptr->string)))


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
  (jna/ensure-type IntByReference item))


(defn ptr-ptr
  ^PointerByReference [item]
  (jna/ensure-ptr-ptr item))

(defn long-ptr
  ^LongByReference [item]
  (jna/ensure-type LongByReference item))



(defn checknil
  ^Pointer [value]
  (if (instance? Pointer value)
    (checknil (Pointer/nativeValue value))
    (if (= 0 (long value))
      (throw (ex-info "Pointer value is nil"
                      {}))
      (Pointer. value))))


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
    (-> (DLPack$DLTensor. (.getValue retval-ptr))
        resource/track)))



(make-tvm-jna-fn TVMFuncListGlobalNames
                 "List the global names"
                 Integer
                 [num-fns int-ptr]
                 [fn-names ptr-ptr])


(defn- array-str-data->string-vec
  [^IntByReference num-fields-ary ^PointerByReference fields]
  (let [base-address (Pointer/nativeValue (.getValue fields))]
    (->> (range (.getValue num-fields-ary))
         (map (fn [name-idx]
                (let [new-ptr (-> (+ (* (long name-idx) Native/POINTER_SIZE)
                                     base-address)
                                  (Pointer.))
                      char-ptr (case Native/POINTER_SIZE
                                 8 (.getLong new-ptr 0)
                                 4 (.getInt new-ptr 0))]
                  (jna/variable-byte-ptr->string (Pointer. char-ptr)))))
         sort
         vec)))


(def global-function-names
  (memoize
   (fn []
     (let [int-data (IntByReference.)
           fn-names (PointerByReference.)]
       (check-call (TVMFuncListGlobalNames int-data fn-names))
       (array-str-data->string-vec int-data fn-names)))))


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


(make-tvm-jna-fn TVMFuncCall
                 "Call a tvm function"
                 Integer
                 [fn-handle checknil
                  arg_values checknil
                  type_codes checknil
                  num_args int
                  ret_val long-ptr
                  ret_type_code int-ptr])


(defn arg-list->tvm-args
 [args]
  (let [num-args (count args)
        arg-vals (dtype-jna/make-typed-pointer :int64 num-args)
        arg-types (dtype-jna/make-typed-pointer :int32 num-args)]
    (->> args
         (map-indexed (fn [idx arg]
                        (let [[long-val dtype] (->tvm-value arg)]
                          (dtype/set-value! arg-vals idx long-val)
                          (dtype/set-value! arg-types idx (keyword->tvm-datatype dtype)))))
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
  (println (format "Failed to map value type %s" val-type-kwd))
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


(defn call-function
  [^Pointer tvm-fn & args]
  (let [fn-ret-val
        (resource/with-resource-context
          (let [retval (LongByReference.)
                rettype (IntByReference.)
                [tvm-args arg-types n-args] (arg-list->tvm-args args)]
            (check-call
             (TVMFuncCall tvm-fn
                          tvm-args arg-types n-args
                          retval rettype))
            [(.getValue retval) (tvm-datatype->keyword-nothrow (.getValue rettype))]))]
    (apply tvm-value->jvm fn-ret-val)))



(defmulti get-extended-node-value
  "Override this to enable type-specific lookups into nodes."
  (fn [node-handle item-key]
    (:tvm-type-kwd (:data node-handle))))


(defmethod get-extended-node-value :default
  [& args]
  nil)


(make-tvm-jna-fn TVMNodeListAttrNames
                 "List the node attributes"
                 Integer
                 [node-handle checknil]
                 [out_size int-ptr]
                 [out_array ptr-ptr])


(defn get-node-fields
  [^Pointer handle]
  (let [fields (PointerByReference.)
        num-fields (IntByReference.)]
    (check-call (TVMNodeListAttrNames handle num-fields fields))
    (array-str-data->string-vec num-fields fields)))


(make-tvm-jna-fn TVMNodeGetAttr
                 "Get a node attribute by name"
                 Integer
                 [node-handle checknil]
                 [key jna/string->ptr]
                 [out_value long-ptr]
                 [out_type_code int-ptr]
                 [out_success int-ptr])


(defn get-node-field
  [^Pointer handle field-name]
  (let [out-tvm-val (LongByReference.)
        out-type-code (IntByReference.)
        out-success (IntByReference.)
        field-name (cond
                     (string? field-name)
                     field-name
                     (keyword? field-name)
                     (name field-name)
                     :else
                     (throw (ex-info "Unrecognized field name type"
                                     {:field-name field-name})))]
    (check-call (TVMNodeGetAttr handle field-name
                                out-tvm-val
                                out-type-code
                                out-success))
    (if (= 1 (.getValue out-success))
      (tvm-value->jvm (.getValue out-tvm-val)
                      (tvm-datatype->keyword-nothrow (.getValue out-type-code)))
      nil)))




(p/def-map-type NodeHandle [^Pointer tvm-jcpp-handle fields data]
  (get [this key default-value]
       (or
        (condp = key
          :tvm-jcpp-handle tvm-jcpp-handle
          :data data
          (if-let [retval (get data key default-value)]
            retval
            (if (fields key)
              (get-node-field tvm-jcpp-handle key)
              (get-extended-node-value this key))))
        default-value))
  (assoc [this key value]
         (if (= key :tvm-jcpp-handle)
           (NodeHandle. value fields data)
           (NodeHandle. tvm-jcpp-handle fields (assoc data key value))))
  (dissoc [this key]
          (NodeHandle. tvm-jcpp-handle fields (dissoc data key)))
  (keys [this]
        (set (concat [:tvm-jcpp-handle] (keys data) fields)))
  (meta [this]
        (meta data))
  (with-meta [this meta]
    (NodeHandle. tvm-jcpp-handle fields (with-meta data meta)))
  ;;There is no equivalence-type system.  Two node handles are equal, equivalent
  ;;if they point to the same raw-object.  Else they aren't.  Additional data
  ;;saved on each on can't matter for the API to work correctly.
  (hasheq [this]
          (.hashCode this))
  (equiv [this other]
         (.equals this other))
  (equals [a b]
          (= (.hashCode a) (.hashCode b)))
  (hashCode [this]
            (call-function (name->global-function "_raw_ptr") this))
  (toString [this]
            (assoc
             (->> (keys this)
                  (map (fn [k]
                         [k (get this k)]))
                  (into {}))
             :raw-ptr (.hashCode this))))


(make-tvm-jna-fn TVMNodeGetTypeIndex
                 "Get the type index of a node."
                 Integer
                 [node-hdl checknil]
                 [out_index int-ptr])


(defn- get-node-type-index
  [^Pointer handle]
  (let [node-type-data (IntByReference.)]
    (check-call
     (TVMNodeGetTypeIndex handle node-type-data))
    (.getValue node-type-data)))


(make-tvm-jna-fn TVMNodeTypeKey2Index
                 "Convert a type name to a type index."
                 Integer
                 [type_key jna/string->ptr]
                 [out_index int-ptr])


(defn- node-type-name->index
  "Convert a node type name to an index."
  [^String type-name]
  (let [int-data (IntByReference.)]
    (check-call (TVMNodeTypeKey2Index type-name int-data))
    (.getValue int-data)))


(def node-type-index->keyword
  (memoize
   (fn [type-index]
     (->> node-type-name->keyword-map
          (map (fn [[type-name keyword]]
                 (when (= type-index (node-type-name->index type-name))
                   keyword)))
          (remove nil?)
          first))))


(defn construct-node
  ^NodeHandle [^Pointer node-handle]
  (let [fields (->> (get-node-fields node-handle)
                    (map keyword)
                    set)
        type-index (get-node-type-index node-handle)
        type-kwd (node-type-index->keyword type-index)]
    (NodeHandle. node-handle fields {:tvm-type-index type-index
                                     :tvm-type-kwd type-kwd})))




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
