(ns tvm-clj.core
  (:require [clojure.reflect :as reflect]
            [tech.javacpp-datatype :as jcpp-dtype]
            [think.resource.core :as resource]
            [clojure.set :as c-set]
            [tvm-clj.base :as base]
            [tech.datatype.base :as dtype]
            [potemkin :as p])
  (:import [tvm_clj.tvm runtime runtime$TVMFunctionHandle runtime$TVMValue
            runtime$NodeHandle runtime$TVMModuleHandle runtime$DLTensor
            runtime$TVMStreamHandle]
           [java.util ArrayList]
           [org.bytedeco.javacpp PointerPointer BytePointer Pointer]
           [java.lang.reflect Field]
           [tvm_clj.base ArrayHandle]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(def ^:dynamic fn-name "")


(defmacro check-call
  [& body]
  `(let [ret# (int (do ~@body))]
     (when-not (= 0 ret#)
       (throw (ex-info "Error during TVM call:"
                       {:error-string (variable-byte-ptr->string (runtime/TVMGetLastError))
                        :fn-name fn-name})))))


(defn unsafe-read-byte
  [^BytePointer byte-ary ^long idx]
  (.set ^Field jcpp-dtype/capacity-field byte-ary (inc idx))
  (.set ^Field jcpp-dtype/limit-field byte-ary (inc idx))
  (let [retval (.get byte-ary idx)]
    retval))


(defn variable-byte-ptr->string
  [^BytePointer byte-ary]
  (String. ^"[B"
           (into-array Byte/TYPE
                       (take-while #(not= % 0)
                                   (map #(unsafe-read-byte
                                          byte-ary %)
                                        (range))))))


(declare call-function)
(declare name->global-function)


;;Because you can't override hashcode on records and multiple nodehandles can point
;;to the same node, we have to create our own extensible map type for node handles
(p/def-map-type NodeHandle [^runtime$NodeHandle tvm-jcpp-handle data]
  (get [this key default-value]
       (if (= key :tvm-jcpp-handle)
         tvm-jcpp-handle
         (get data key default-value)))
  (assoc [this key value]
         (if (= key :tvm-jcpp-handle)
           (NodeHandle. value data)
           (NodeHandle. tvm-jcpp-handle (assoc data key value))))
  (dissoc [this key]
          (NodeHandle. tvm-jcpp-handle (dissoc data key)))
  (keys [this]
        (vec (concat [:tvm-jcpp-handle] (keys data))))
  (meta [this]
        (meta data))
  (with-meta [this meta]
    (NodeHandle. tvm-jcpp-handle (with-meta data meta)))


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


(extend-protocol resource/PResource
  NodeHandle
  (release-resource [node]
    (runtime/TVMNodeFree (.tvm-jcpp-handle node)))
  runtime$TVMModuleHandle
  (release-resource [module]
    (runtime/TVMModFree module))
  runtime$TVMFunctionHandle
  (release-resource [tvm-fn]
    (runtime/TVMFuncFree tvm-fn))
  ArrayHandle
  (release-resource [tvm-dev-ar]
    (runtime/TVMArrayFree (.tvm-jcpp-handle tvm-dev-ar))))



(defn- array-str-data->string-vec
  [^ints num-names ^PointerPointer name-data-ptr]
  (->> (range (aget num-names 0))
       (map (fn [^long idx]
              (let [name-data-ptr (BytePointer. (.get name-data-ptr idx))]
                (variable-byte-ptr->string name-data-ptr))))
       sort
       vec))


(def global-function-names
  (memoize
   (fn []
     (let [num-names (int-array [1])
           name-data-ptr (PointerPointer. 1)]
       (runtime/TVMFuncListGlobalNames num-names name-data-ptr)
       (let [retval (array-str-data->string-vec num-names name-data-ptr)]
         (resource/release name-data-ptr)
         retval)))))


(def name->global-function
  "The function returned should not be freed"
  (memoize
   (fn [^String fn-name]
     (let [retval (runtime$TVMFunctionHandle.)]
       (check-call
        (runtime/TVMFuncGetGlobal fn-name retval))
       (when (= 0 (.address retval))
         (throw (ex-info "Failed to find global function"
                         {:fn-name fn-name})))
       retval))))

(declare tvm-datatype->keyword)
(declare tvm-value->jvm)
(declare tvm-array->jvm)
(declare tvm-map->jvm)


(defn- tvm-value->long
  ^long [^runtime$TVMValue value]
  (let [bb (.asByteBuffer value)
        lb (.asLongBuffer bb)]
    (.get lb 0)))


(defn- expand-node-field
  [^NodeHandle node-handle ^String field-name]
  (let [retval (resource/with-resource-context
                 (let [field-val (runtime$TVMValue. 1)
                       field-type (int-array 1)
                       success (int-array 1)]
                   (runtime/TVMNodeGetAttr ^runtime$NodeHandle (.tvm-jcpp-handle node-handle)
                                           field-name
                                           field-val
                                           field-type
                                           success)
                   [(tvm-value->long field-val) (tvm-datatype->keyword (aget field-type 0))]))]
    (tvm-value->jvm retval)))


(defn- list-node-fields
  [^NodeHandle node]
  (let [runtime-handle (.tvm-jcpp-handle node)
        fields (PointerPointer. 1)
        num-fields-ary (int-array 1)]
    (runtime/TVMNodeListAttrNames ^runtime$NodeHandle runtime-handle num-fields-ary fields)
    (let [retval (array-str-data->string-vec num-fields-ary fields)]
      (resource/release fields)
      retval)))


(defn- get-node-type-index
  [^NodeHandle node]
  (let [node-type-data (int-array 1)]
    (runtime/TVMNodeGetTypeIndex ^runtime$NodeHandle (.tvm-jcpp-handle node) node-type-data)
    (aget node-type-data 0)))


(defn- get-type-key-for-name
  [^String type-name]
  (let [int-data (int-array 1)]
    (runtime/TVMNodeTypeKey2Index type-name int-data)
    (aget int-data 0)))

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
  [^NodeHandle node]
  (expression-set (:tvm-type-kwd node)))


(def node-type-name->index
  (memoize get-type-key-for-name))

;;May return nil
(def node-type-index->keyword
  (memoize
   (fn [type-index]
     (->> node-type-name->keyword-map
          (map (fn [[type-name keyword]]
                 (when (= type-index (node-type-name->index type-name))
                   keyword)))
          (remove nil?)
          first))))


(defn- make-node-handle
  "Any time this is called the object should be freed eventually:
https://github.com/dmlc/tvm/issues/918"
  ^runtime$NodeHandle [^long handle-value]
  (let [runtime-handle (runtime$NodeHandle.)
        retval (NodeHandle. runtime-handle {})]
    (.set ^Field jcpp-dtype/address-field runtime-handle handle-value)
    (let [node-type (get-node-type-index retval)]
      (assoc retval
             :tvm-type-index node-type
             :tvm-type-kwd (node-type-index->keyword node-type)))))


(declare unpack-node-fields)


(defn unpack-node-field
  [node-field-val & {:keys [recurse]
                     :or {recurse true}}]

  (let [local-unpack (if recurse
                       unpack-node-fields
                       identity)]
    (cond
      (= :array (get node-field-val :tvm-type-kwd))
      (let [ary-data (tvm-array->jvm node-field-val)]
        (resource/release node-field-val)
        (mapv local-unpack ary-data))
      (= :map (get node-field-val :tvm-type-kwd))
      (let [map-data (tvm-map->jvm node-field-val)]
        (resource/release node-field-val)
        (->> map-data
             (map (fn [[k v]]
                    [(local-unpack k) (local-unpack v)]))
             (into {})))
      (instance? NodeHandle node-field-val)
      (local-unpack node-field-val)
      :else
      node-field-val)))


(defn unpack-node-fields
  [^NodeHandle node & {:keys [recurse]
                       :or {recurse true}}]
  (->> (list-node-fields node)
       (reduce (fn [retval field-name]
                 (let [node-field-val (expand-node-field retval field-name)]
                   (assoc retval (keyword field-name)
                          (unpack-node-field node-field-val :recurse recurse))))
               node)))


(declare tvm-array)
(declare tvm-map)


(extend-protocol base/PJVMTypeToTVMValue
  Double
  (jvm->tvm-value [value] [(Double/doubleToLongBits (double value)) runtime/kDLFloat])
  Float
  (jvm->tvm-value [value] [(Double/doubleToLongBits (double value)) runtime/kDLFloat])
  Byte
  (jvm->tvm-value [value] [(long value) runtime/kDLInt])
  Short
  (jvm->tvm-value [value] [(long value) runtime/kDLInt])
  Integer
  (jvm->tvm-value [value] [(long value) runtime/kDLInt])
  Long
  (jvm->tvm-value [value] [(long value) runtime/kDLInt])
  Boolean
  (jvm->tvm-value [value] [(if value
                             (long 1)
                             (long 0)) runtime/kDLInt])
  String
  (jvm->tvm-value [value]
    (let [pb (BytePointer. value "ASCII")]
      (resource/track pb)
      [(.address pb) runtime/kStr]))

  NodeHandle
  (jvm->tvm-value [value]
    [(.address ^runtime$NodeHandle (.tvm-jcpp-handle ^NodeHandle value)) runtime/kNodeHandle])

  ArrayHandle
  (jvm->tvm-value [value]
    [(.address ^runtime$DLTensor (.tvm-jcpp-handle ^ArrayHandle value)) runtime/kArrayHandle])

  Object
  (jvm->tvm-value [value]
    (cond
      (sequential? value)
      (base/jvm->tvm-value (apply tvm-array value))
      (map? value)
      (base/jvm->tvm-value (apply tvm-map (->> (seq value)
                                               (apply concat))))
      (nil? value)
      [0 runtime/kNull]))

  nil
  (jvm->tvm-value [value]
    [0 runtime/kNull]))


(defn arg-list->tvm-args
 [args]
  (let [num-args (count args)
        retval (runtime$TVMValue. num-args)
        bb (.asByteBuffer retval)
        lb (.asLongBuffer bb)
        type-codes (int-array num-args)]
    (resource/track retval)
    (->> args
         (map-indexed (fn [idx arg]
                        (let [[long-val dtype] (base/jvm->tvm-value arg)]
                          (.put lb (int idx) (long long-val))
                          (aset type-codes idx (int dtype)))))
         dorun)
    [retval type-codes]))


(def tvm-datatype->keyword-map
  {runtime/kDLInt :int
   runtime/kDLUInt :uint
   runtime/kDLFloat :float
   runtime/kHandle :handle
   runtime/kNull :null
   runtime/kTVMType :tvm-type
   runtime/kTVMContext :tvm-context
   runtime/kArrayHandle :array-handle
   runtime/kNodeHandle :node-handle
   runtime/kModuleHandle :module-handle
   runtime/kFuncHandle :func-handle
   runtime/kStr :string
   runtime/kBytes :bytes
   })


(defn tvm-datatype->keyword
  [^long tvm-datatype-int]
  (get tvm-datatype->keyword-map tvm-datatype-int))


(def keyword->tvm-datatype-map
  (c-set/map-invert tvm-datatype->keyword-map))


(defn keyword->tvm-datatype
  ^long [kwd]
  (long (keyword->tvm-datatype kwd)))


(declare make-module-handle)


(defn tvm-value->jvm
  "Attempts to coerce the tvm value into the jvm.  Failures
result in a returned map container a value for the key:
:tvm->jvm-failure

This is in order to ensure that, for instance, deserialization of a node's fields
allows for a sane recovery mechanism and doesn't lose those field values."
  [[long-val val-type-kwd]]
  (let [long-val (long long-val)]
    (try
      (condp = val-type-kwd
        :int
        long-val
        :uint
        long-val
        :float
        (Double/longBitsToDouble long-val)
        :string
        (let [bp (BytePointer.)]
          (.set ^Field jcpp-dtype/address-field bp long-val)
          (variable-byte-ptr->string bp))
        :node-handle
        (make-node-handle long-val)
        :module-handle
        (make-module-handle long-val))
      (catch Throwable e
        {:tvm->jvm-failure e
         :val-keyword val-type-kwd}))))


(defn call-function
  [^runtime$TVMFunctionHandle tvm-fn & args]
  (let [fn-ret-val
        (resource/with-resource-context
          (let [^runtime$TVMValue retval (resource/track (runtime$TVMValue. 1))
                rettype (int-array 1)
                [tvm-args arg-types] (arg-list->tvm-args args)]
            (check-call
             (runtime/TVMFuncCall tvm-fn
                                  ^runtime$TVMValue tvm-args
                                  ^ints arg-types
                                  (count arg-types)
                                  retval
                                  rettype))
            [(tvm-value->long retval) (tvm-datatype->keyword-map (aget rettype 0))]))]
    (tvm-value->jvm fn-ret-val)))


(defn tvm-array
  "Called when something like a shape needs to be passed into a tvm function.  Most users will not need to call this
explicitly; it is done for you."
  [& args]
  (apply call-function (name->global-function "_Array") args))


(defn tvm-map
  "Call like hash-map except all keys must be node handles.  Most users will not need to call this explicitly"
  [& args]
  (when-not (= 0 (rem (count args)
                      2))
    (throw (ex-info "Map fn call must have even arg count"
                    {:args args})))
  (with-bindings {#'fn-name "_Map"}
    (apply call-function (name->global-function "_Map") args)))


(defn tvm-array->jvm
  [tvm-array-node]
  (->> (range (call-function (name->global-function "_ArraySize") tvm-array-node))
       (mapv #(call-function (name->global-function "_ArrayGetItem") tvm-array-node (int %1)))))


(defn tvm-map->jvm
  [tvm-map-node]
  (->> (call-function (name->global-function "_MapItems") tvm-map-node)
       tvm-array->jvm
       (apply hash-map)))


(defn global-node-function
  "Call a global function that returns a node."
  [fn-name & args]
  (with-bindings {#'fn-name fn-name}
    (-> (apply call-function (name->global-function fn-name) args)
        unpack-node-fields)))

(defn global-function
  [fn-name & args]
  (with-bindings {#'fn-name fn-name}
    (apply call-function (name->global-function fn-name) args)))

(def g-fn global-function)
(def gn-fn global-node-function)

(defn get-node-type
  [node]
  (get node :tvm-type-kwd))


(defn make-module-handle
  ^runtime$TVMModuleHandle [^long module-jcpp-handle]
  (let [retval (runtime$TVMModuleHandle.)]
    (.set ^Field jcpp-dtype/address-field retval module-jcpp-handle)
    (resource/track retval)))


(defn get-module-function
  [^runtime$TVMModuleHandle module ^String fn-name & {:keys [query-imports?]}]
  (let [retval (runtime$TVMFunctionHandle.)]
    (check-call (runtime/TVMModGetFunction module fn-name (int (if query-imports? 1 0)) retval))
    (when (= 0 (.address retval))
      (throw (ex-info "Could not find module function"
                      {:fn-name fn-name})))
    retval))


(def datatype->tvm-datatype-data-map
  {:int8 {:dtype-code runtime/kDLInt
          :dtype-bits 8
          :dtype-lanes 1
          :name "int8"}
   :int16 {:dtype-code runtime/kDLInt
           :dtype-bits 16
           :dtype-lanes 1
           :name "int16"}
   :int32 {:dtype-code runtime/kDLInt
           :dtype-bits 32
           :dtype-lanes 1
           :name "int32"}
   :int64 {:dtype-code runtime/kDLInt
           :dtype-bits 64
           :dtype-lanes 1
           :name "int64"}
   :uint8 {:dtype-code runtime/kDLUInt
           :dtype-bits 8
           :dtype-lanes 1
           :name "uint8"}
   :uint16 {:dtype-code runtime/kDLUInt
            :dtype-bits 16
            :dtype-lanes 1
            :name "uint16"}
   :uint32 {:dtype-code runtime/kDLUInt
            :dtype-bits 32
            :dtype-lanes 1
            :name "uint32"}
   :uint64 {:dtype-code runtime/kDLUInt
            :dtype-bits 64
            :dtype-lanes 1
            :name "uint64"}
   :float32 {:dtype-code runtime/kDLFloat
             :dtype-bits 32
             :dtype-lanes 1
             :name "float32"}
   :float64 {:dtype-code runtime/kDLFloat
             :dtype-bits 64
             :dtype-lanes 1
             :name "float64"}})


(defn datatype->tvm-datatype-data
  [datatype]
  (if-let [retval (datatype->tvm-datatype-data-map datatype)]
    retval
    (throw (ex-info "Failed to find tvm datatype for datatype"
                    {:datatype datatype}))))


(def kwd->device-type-map
  {:cpu runtime/kDLCPU
   :llvm runtime/kDLCPU
   :stackvm runtime/kDLCPU
   :cuda runtime/kDLGPU
   :cpu-pinned runtime/kDLCPUPinned
   :opencl runtime/kDLOpenCL
   :metal runtime/kDLMetal
   :vpi runtime/kDLVPI
   :rocm runtime/kDLROCM
   :vulkan runtime/kDLVulkan
   :opengl runtime/kOpenGL
   ;; // Extension DRAM type, used for quickly test extension device
   ;; // The device api can differ depending on the xpu driver registered.
   :ext-dev runtime/kExtDev
   ;; // AddExtraTVMType which is not in DLPack here
   })


(defn device-type->device-type-int
  ^long [device-type]
  (if-let [dev-enum (kwd->device-type-map device-type)]
    dev-enum
    (throw (ex-info "Failed to find device type enum"
                    {:device-type device-type}))))


(defn allocate-device-array
  [shape datatype device-type ^long device-id]
  (let [n-dims (count shape)
        shape-data (long-array n-dims)
        _ (dtype/copy-raw->item! shape shape-data 0)
        retval-ptr (PointerPointer. 1)
        {:keys [dtype-code dtype-bits dtype-lanes]}
        (datatype->tvm-datatype-data datatype)
        retval (runtime$DLTensor.)
        device-type-int (device-type->device-type-int device-type)]
    (check-call
     (runtime/TVMArrayAlloc shape-data (int n-dims) (int dtype-code) (int dtype-bits) (int dtype-lanes)
                            device-type-int device-id retval-ptr))
    (.set ^Field jcpp-dtype/address-field retval (.address (.get retval-ptr 0)))
    (resource/release retval-ptr)
    (resource/track (merge (base/->ArrayHandle retval)
                           {:shape shape
                            :datatype datatype
                            :device-type device-type
                            :device-id device-id}))))


(defn copy-to-array!
  [^Pointer src ^ArrayHandle dest ^long n-bytes]
  (check-call (runtime/TVMArrayCopyFromBytes
               ^runtime$DLTensor (.tvm-jcpp-handle dest)
               src n-bytes)))


(defn copy-from-array!
  [^ArrayHandle src ^Pointer dest ^long n-bytes]
  (check-call (runtime/TVMArrayCopyToBytes
               ^runtime$DLTensor (.tvm-jcpp-handle src)
               dest n-bytes)))


(defn copy-array-to-array!
  [^ArrayHandle src-hdl ^ArrayHandle dst-hdl]
  (check-call (runtime/TVMArrayCopyFromTo
               ^runtime$DLTensor (.tvm-jcpp-handle src-hdl)
               ^runtime$DLTensor (.tvm-jcpp-handle dst-hdl)
               (runtime$TVMStreamHandle.))))
