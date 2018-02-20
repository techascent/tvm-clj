(ns tvm-clj.core
  (:require [clojure.reflect :as reflect]
            [tech.javacpp-datatype :as jcpp-dtype]
            [think.resource.core :as resource]
            [clojure.set :as c-set])
  (:import [tvm_clj.tvm runtime runtime$TVMFunctionHandle runtime$TVMValue]
           [java.util ArrayList]
           [org.bytedeco.javacpp PointerPointer BytePointer]
           [java.lang.reflect Field]))


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

(def global-function-names
  (memoize
   (fn
     []
     (let [num-names (int-array [1])
           name-data-ptr (PointerPointer. 1)]
       (runtime/TVMFuncListGlobalNames num-names name-data-ptr)
       (->> (range (aget num-names 0))
            (map (fn [idx]
                   (let [name-data-ptr (BytePointer. (.get name-data-ptr idx))]
                     (variable-byte-ptr->string name-data-ptr))))
            sort
            vec)))))


(def global-function
  "The function returned should not be freed"
  (memoize
   (fn [^String fn-name]
     (let [retval (runtime$TVMFunctionHandle.)]
       (runtime/TVMFuncGetGlobal fn-name retval)
       retval))))


(defprotocol PJVMTypeToTVMValue
  (jvm->tvm-value [jvm-type]))


(defrecord NodeHandle [^long handle])


(declare tvm-array)


(extend-protocol PJVMTypeToTVMValue
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
    [(:handle value) runtime/kNodeHandle])
  clojure.lang.Sequential
  (jvm->tvm-value [value]
    (jvm->tvm-value (apply tvm-array value))))


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
                        (let [[long-val dtype] (jvm->tvm-value arg)]
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
  [^runtime$TVMValue val val-type-kwd]
  (let [bb (.asByteBuffer val)
        lb (.asLongBuffer bb)
        long-val (.get lb 0)]
    (condp = val-type-kwd
      :int
      long-val
      :uint
      long-val
      :float
      (Double/longBitsToDouble long-val)
      :node-handle
      (->NodeHandle long-val))))


(defn call-function
  [^runtime$TVMFunctionHandle tvm-fn & args]
  (let [retval (resource/track (runtime$TVMValue. 1))
        rettype (int-array 1)
        [tvm-args arg-types] (arg-list->tvm-args args)]
    (check-call
     (runtime/TVMFuncCall tvm-fn
                          ^runtime$TVMValue tvm-args
                          ^ints arg-types
                          (count arg-types)
                          retval
                          rettype))
    (tvm-value->jvm retval (tvm-datatype->keyword-map (aget rettype 0)))))


(defn variable
  "Create a simple variable.  Returns a node handle"
  [^String name & {:keys [type-str]
                   :or {type-str "int32"}}]
  (call-function (global-function "_Var") name type-str))


(defn tvm-array
  [& args]
  (apply call-function (global-function "_Array") args))


(defn placeholder
  [shape & {:keys [dtype name]
            :or {dtype "float32"
                 name "placeholder"}}]
  (let [shape (if-not (instance? clojure.lang.Seqable shape)
                [shape]
                shape)]
    (call-function (global-function "_Placeholder") shape dtype name)))
