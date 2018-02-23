(ns tvm-clj.core
  (:require [clojure.reflect :as reflect]
            [tech.javacpp-datatype :as jcpp-dtype]
            [think.resource.core :as resource]
            [clojure.set :as c-set]
            [tvm-clj.base :as base])
  (:import [tvm_clj.tvm runtime runtime$TVMFunctionHandle runtime$TVMValue
            runtime$NodeHandle]
           [java.util ArrayList]
           [org.bytedeco.javacpp PointerPointer BytePointer]
           [java.lang.reflect Field]
           [tvm_clj.base NodeHandle]))


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
        retval (base/->NodeHandle runtime-handle)]
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


(extend-type NodeHandle
  resource/PResource
  (release-resource [node]
    (runtime/TVMNodeFree (.tvm-jcpp-handle node))))


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

  Object
  (jvm->tvm-value [value]
    (cond
      (sequential? value)
      (base/jvm->tvm-value (apply tvm-array value))
      (map? value)
      (cond
        (instance? NodeHandle value)
        [(.address ^runtime$NodeHandle (.tvm-jcpp-handle ^NodeHandle value)) runtime/kNodeHandle]
        :else
        (base/jvm->tvm-value (apply tvm-map (->> (seq value)
                                                 (apply concat)))))
      (nil? value)
      [0 runtime/kNull])))


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
        (make-node-handle long-val))
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
