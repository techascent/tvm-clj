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


(def global-function
  "The function returned should not be freed"
  (memoize
   (fn [^String fn-name]
     (let [retval (runtime$TVMFunctionHandle.)]
       (runtime/TVMFuncGetGlobal fn-name retval)
       retval))))

(declare tvm-datatype->keyword)
(declare tvm-value->jvm)
(declare tvm-array->jvm)


(defn- expand-node-field
  [^NodeHandle node-handle ^String field-name]
  (let [field-val (runtime$TVMValue. 1)
        field-type (int-array 1)
        success (int-array 1)]
    (runtime/TVMNodeGetAttr ^runtime$NodeHandle (.tvm-jcpp-handle node-handle)
                            field-name
                            field-val
                            field-type
                            success)
    (tvm-value->jvm field-val (tvm-datatype->keyword (aget field-type 0)))))

(defn- list-node-fields
  [^NodeHandle node]
  (let [runtime-handle (.tvm-jcpp-handle node)
        fields (PointerPointer. 1)
        num-fields-ary (int-array 1)]
    (runtime/TVMNodeListAttrNames ^runtime$NodeHandle runtime-handle num-fields-ary fields)
    (let [retval (array-str-data->string-vec num-fields-ary fields)]
      (resource/release fields)
      retval)))


(defn- get-node-type
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
  {"Array" :array})


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
  ^runtime$NodeHandle [^long handle-value]
  (let [runtime-handle (runtime$NodeHandle.)
        retval (base/->NodeHandle runtime-handle)]
    (.set ^Field jcpp-dtype/address-field runtime-handle handle-value)
    (let [node-type (get-node-type retval)]
      (assoc retval
             :tvm-type-index node-type
             :tvm-type-kwd (node-type-index->keyword node-type)))))


(defn- unpack-node-fields
  [^NodeHandle node]
  (->> (list-node-fields node)
       (reduce (fn [retval field-name]
                 (let [node-field-val (expand-node-field retval field-name)]
                   (assoc retval (keyword field-name)
                          (cond
                            (= :array (get node-field-val :tvm-type-kwd))
                            (tvm-array->jvm node-field-val)
                            (instance? NodeHandle node-field-val)
                            (unpack-node-fields node-field-val)
                            :else
                            node-field-val))))
               node)))


(extend-type NodeHandle
  resource/PResource
  (release-resource [node]
    (runtime/TVMNodeFree (.tvm-jcpp-handle node))))


(declare tvm-array)


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
  String
  (jvm->tvm-value [value]
    (let [pb (BytePointer. value "ASCII")]
      (resource/track pb)
      [(.address pb) runtime/kStr]))
  NodeHandle
  (jvm->tvm-value [value]
    [(.address ^runtime$NodeHandle (.tvm-jcpp-handle value)) runtime/kNodeHandle])
  clojure.lang.Sequential
  (jvm->tvm-value [value]
    (base/jvm->tvm-value (apply tvm-array value))))


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


(defmacro check-call
  [& body]
  `(let [ret# (int (do ~@body))]
     (when-not (= 0 ret#)
       (throw (ex-info "Error during TVM call:"
                       {:error-string (variable-byte-ptr->string (runtime/TVMGetLastError))})))))

(defn tvm-value->jvm
  "Attempts to coerce the tvm value into the jvm.  Failures
result in a returned map container a value for the key:
:tvm->jvm-failure

This is in order to ensure that, for instance, deserialization of a node's fields
allows for a sane recovery mechanism and doesn't lose those field values."
  [^runtime$TVMValue val val-type-kwd]
  (let [bb (.asByteBuffer val)
        lb (.asLongBuffer bb)
        long-val (.get lb 0)]
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
  (let [retval (resource/with-resource-context
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
                   (tvm-value->jvm retval (tvm-datatype->keyword-map (aget rettype 0)))))]
    retval))


(defn variable
  "Create a simple variable.  Returns a node handle"
  [^String name & {:keys [type-str]
                   :or {type-str "int32"}}]
  (resource/track
   (unpack-node-fields
    (call-function (global-function "_Var") name type-str))))


(defn- tvm-array
  "Called when something like a shape needs to be passed into a tvm function"
  [& args]
  (resource/track
   (apply call-function (global-function "_Array") args)))


(defn placeholder
  [shape & {:keys [dtype name]
            :or {dtype "float32"
                 name "placeholder"}}]
  (let [shape (if-not (instance? clojure.lang.Seqable shape)
                [shape]
                shape)]
    (resource/track
     (unpack-node-fields
      (call-function (global-function "_Placeholder") shape dtype name)))))


(defn- tvm-array->jvm
  [tvm-array-node]
  (->> (range (call-function (global-function "_ArraySize") tvm-array-node))
       (mapv #(call-function (global-function "_ArrayGetItem") tvm-array-node (int %1)))))
