(ns tech.compute.tvm.host-buffer
  "Host buffer datatype that supports unsigned datatypes"
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.datatype.base :as dtype]
            [tech.datatype.marshal :as marshal]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.datatype.core]
            [clojure.core.matrix.protocols :as mp]
            [think.resource.core :as resource]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.base :as tvm-comp-base])
  (:import [org.bytedeco.javacpp BytePointer ShortPointer
            IntPointer LongPointer FloatPointer DoublePointer
            Pointer]
           [java.nio Buffer ByteBuffer ShortBuffer IntBuffer
            LongBuffer FloatBuffer DoubleBuffer]))


(defmacro unsigned->jvm
  [src-dtype val]
  (condp = src-dtype
    :uint8 `(bit-and (unchecked-short ~val) 0xFF)
    :uint16 `(bit-and (unchecked-int ~val) 0xFFFF)
    :uint32 `(bit-and (unchecked-long ~val) 0xFFFFFFFF)
    :uint64 `(bit-and (unchecked-long ~val) 0xFFFFFFFFFFFFFFFFF)
    `~val))


(defmacro jvm->unsigned
  [dst-dtype val]
  (condp = dst-dtype
    :uint8 `(unchecked-byte ~val)
    :uint16 `(unchecked-short ~val)
    :uint32 `(unchecked-int ~val)
    :uint64 `(unchecked-long ~val)
    `~val))


(defmacro unsigned-cast-macro
  [dtype]
  {:from `(fn [val#]
            (unsigned->jvm ~dtype val#))
   :to `(fn [val#]
          (jvm->unsigned ~dtype val#))})


(def unsigned-types [:uint8 :uint16 :uint32 :uint64])
(def unsigned-type-set (set unsigned-types))


(defmacro unsigned-scalar-conversion-table-macro
  []
  (->> unsigned-types
       (map (fn [dtype]
              [dtype
               `(unsigned-cast-macro ~dtype)]))
       (into {})))


(def unsigned-scalar-conversion-table
  (unsigned-scalar-conversion-table-macro))


(defn jcpp-pointer-sub-buffer
  [^Pointer ptr offset length]
  (-> (jcpp-dtype/offset-pointer ptr offset)
      (jcpp-dtype/set-pointer-limit-and-capacity length)))


(defn jcpp-pointer-alias?
  [^Pointer lhs ^Pointer rhs]
  (= (.address lhs)
     (.address rhs)))

(defn jcpp-pointer-byte-length
  [^Pointer ptr]
  (* (dtype/ecount ptr)
     (dtype/datatype-size-map
      (dtype/get-datatype ptr))))


(defn jcpp-pointer-partial-alias?
  [^Pointer lhs ^Pointer rhs]
  (let [l-start (.address lhs)
        r-start (.address rhs)
        l-end (+ l-start (jcpp-pointer-byte-length lhs))
        r-end (+ r-start (jcpp-pointer-byte-length rhs))]
    (or (and (>= r-start l-start)
             (< r-start l-end))
        (and (>= l-start r-start)
             (< l-start r-end)))))


(defrecord HostBuffer [^Pointer ptr tvm-dtype]
  dtype/PDatatype
  (get-datatype [this] tvm-dtype)

  dtype/PAccess
  (set-value! [item offset value]
    (let [conv-fn (get-in unsigned-scalar-conversion-table [tvm-dtype :to])
          value (if conv-fn (conv-fn value) value)]
      (dtype/set-value! ptr offset value)))
  (set-constant! [item offset value elem-count]
    (let [conv-fn (get-in unsigned-scalar-conversion-table [tvm-dtype :to])
          value (if conv-fn (conv-fn value) value)]
      (dtype/set-constant! ptr offset value elem-count)))
  (get-value [item offset]
    (let [conv-fn (get-in unsigned-scalar-conversion-table [tvm-dtype :from])]
      (if conv-fn
        (conv-fn (dtype/get-value ptr offset))
        (dtype/get-value ptr offset))))

  mp/PElementCount
  (element-count [item] (mp/element-count ptr))

  resource/PResource
  (release-resource [this]
    (resource/release-resource ptr))

  marshal/PContainerType
  (container-type [this] :tvm-host-buffer)

  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (->HostBuffer (jcpp-pointer-sub-buffer ptr offset length)
                  tvm-dtype))
  (alias? [lhs rhs]
    (jcpp-pointer-alias? lhs))
  (partially-alias? [lhs rhs]
    (jcpp-pointer-partial-alias? lhs rhs))

  tvm-comp-base/PConvertToTVM
  (->tvm [_] ptr))


(defn host-buffer->byte-ptr
  ^BytePointer [^HostBuffer host-buf]
  (.ptr host-buf))

(defn host-buffer->short-ptr
  ^ShortPointer [^HostBuffer host-buf]
  (.ptr host-buf))

(defn host-buffer->int-ptr
  ^IntPointer [^HostBuffer host-buf]
  (.ptr host-buf))

(defn host-buffer->Long-ptr
  ^LongPointer [^HostBuffer host-buf]
  (.ptr host-buf))

(defn host-buffer->float-ptr
  ^FloatPointer [^HostBuffer host-buf]
  (.ptr host-buf))

(defn host-buffer->double-ptr
  ^DoublePointer [^HostBuffer host-buf]
  (.ptr host-buf))


(defmacro host-buffer->ptr
  [dtype host-buf]
  (condp = dtype
    :int8 `(host-buffer->byte-ptr ~host-buf)
    :uint8 `(host-buffer->byte-ptr ~host-buf)
    :int16 `(host-buffer->short-ptr ~host-buf)
    :uint16 `(host-buffer->short-ptr ~host-buf)
    :int32 `(host-buffer->int-ptr ~host-buf)
    :uint32 `(host-buffer->int-ptr ~host-buf)
    :int64 `(host-buffer->long-ptr ~host-buf)
    :uint64 `(host-buffer->long-ptr ~host-buf)
    :float32 `(host-buffer->float-ptr ~host-buf)
    :float64 `(host-buffer->double-ptr ~host-buf)))


(defn host-buffer->byte-nio-buffer
  ^ByteBuffer [^HostBuffer host-buf]
  (jcpp-dtype/as-buffer (.ptr host-buf)))

(defn host-buffer->short-nio-buffer
  ^ShortBuffer [^HostBuffer host-buf]
  (jcpp-dtype/as-buffer (.ptr host-buf)))

(defn host-buffer->int-nio-buffer
  ^IntBuffer [^HostBuffer host-buf]
  (jcpp-dtype/as-buffer (.ptr host-buf)))

(defn host-buffer->long-nio-buffer
  ^LongBuffer [^HostBuffer host-buf]
  (jcpp-dtype/as-buffer (.ptr host-buf)))

(defn host-buffer->float-nio-buffer
  ^FloatBuffer [^HostBuffer host-buf]
  (jcpp-dtype/as-buffer (.ptr host-buf)))

(defn host-buffer->double-nio-buffer
  ^DoubleBuffer [^HostBuffer host-buf]
  (jcpp-dtype/as-buffer (.ptr host-buf)))


(defmacro host-buffer->nio-buffer
  [dtype host-buf]
  (condp = dtype
    :int8 `(host-buffer->byte-nio-buffer ~host-buf)
    :uint8 `(host-buffer->byte-nio-buffer ~host-buf)
    :int16 `(host-buffer->short-nio-buffer ~host-buf)
    :uint16 `(host-buffer->short-nio-buffer ~host-buf)
    :int32 `(host-buffer->int-nio-buffer ~host-buf)
    :uint32 `(host-buffer->int-nio-buffer ~host-buf)
    :int64 `(host-buffer->long-nio-buffer ~host-buf)
    :uint64 `(host-buffer->long-nio-buffer ~host-buf)
    :float32 `(host-buffer->float-nio-buffer ~host-buf)
    :float64 `(host-buffer->double-nio-buffer ~host-buf)))


;;Build out the marshalling conversion table but only to array and back.
;;nio buffers are just not worth it.

(def direct-conversion-pairs
  [[:uint8 :int8]
   [:uint16 :int16]
   [:uint32 :int32]
   [:uint64 :int64]])

(def direct-conversion-map
  (->> direct-conversion-pairs
       (mapcat (fn [[x y]]
                 [[x y]
                  [y x]]))
       (into {})))

(def full-datatype-list (vec (concat dtype/datatypes unsigned-types)))

(def full-conversion-sequence
  (->> (for [src-dtype full-datatype-list
             dst-dtype full-datatype-list]
         [src-dtype dst-dtype])))


(defn signed-datatype? [dtype] (not (unsigned-type-set dtype)))

(defn direct-conversion?
  [src-dtype dst-dtype]
  (or (= src-dtype dst-dtype)
      (and (signed-datatype? src-dtype)
           (signed-datatype? dst-dtype))
      (= (direct-conversion-map src-dtype) dst-dtype)))

(defmacro array->host-buffer-conversion
  [src-dtype dst-dtype]
  (if (direct-conversion? src-dtype dst-dtype)
    `(fn [src-ary# src-offset# dst-buf# dst-offset# elem-count#]
       (marshal/copy! src-ary# src-offset#
                      (host-buffer->nio-buffer ~dst-dtype dst-buf#) dst-offset#
                      elem-count#))
    `(fn [src-ary# src-offset# dst-buf# dst-offset# elem-count#]
       (let [elem-count# (long elem-count#)
             src-ary# (marshal/datatype->array-cast-fn ~src-dtype src-ary#)
             src-offset# (long src-offset#)
             dst-buf# (host-buffer->nio-buffer ~dst-dtype dst-buf#)
             dst-offset# (long dst-offset#)]
         (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
                (.put dst-buf# (+ dst-offset# idx#)
                      (jvm->unsigned ~dst-dtype (aget src-ary#
                                                      (+ src-offset# idx#)))))))))


(defmacro host-buffer->array-conversion
  [src-dtype dst-dtype]
  (if (direct-conversion? src-dtype dst-dtype)
    `(fn [src-buf# src-offset# dst-ary# dst-offset# elem-count#]
       (marshal/copy! (host-buffer->nio-buffer ~src-dtype src-buf#) src-offset#
                      dst-ary# dst-offset#
                      elem-count#))
    `(fn [src-buf# src-offset# dst-ary# dst-offset# elem-count#]
       (let [elem-count# (long elem-count#)
             src-buf# (host-buffer->nio-buffer ~src-dtype src-buf#)
             src-offset# (long src-offset#)
             dst-ary# (marshal/datatype->array-cast-fn ~dst-dtype dst-ary#)
             dst-offset# (long dst-offset#)]
         (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
                (aset dst-ary# (+ dst-offset# idx#)
                      (marshal/datatype->cast-fn
                       ~dst-dtype
                       (unsigned->jvm ~src-dtype (.get src-buf#
                                                       (+ src-offset# idx#))))))))))


(defmacro host-buffer->host-buffer-conversion
  [src-dtype dst-dtype]
  (if (direct-conversion? src-dtype dst-dtype)
    `(fn [src-buf# src-offset# dst-buf# dst-offset# elem-count#]
       (marshal/copy! (host-buffer->nio-buffer ~src-dtype src-buf#) src-offset#
                      (host-buffer->nio-buffer ~dst-dtype dst-buf#) dst-offset#
                      elem-count#))
    `(fn [src-buf# src-offset# dst-buf# dst-offset# elem-count#]
       (let [elem-count# (long elem-count#)
             src-buf# (host-buffer->nio-buffer ~src-dtype src-buf#)
             src-offset# (long src-offset#)
             dst-buf# (host-buffer->nio-buffer ~dst-dtype dst-buf#)
             dst-offset# (long dst-offset#)]
         (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
                (.put dst-buf# (+ dst-offset# idx#)
                      (jvm->unsigned
                       ~dst-dtype
                       (unsigned->jvm ~src-dtype
                                      (.get src-buf#
                                            (+ src-offset# idx#))))))))))


(defmacro build-full-conversion
  []
  {[:java-array :tvm-host-buffer]
   (->> full-conversion-sequence
        ;;The arrays can only be the core jvm types
        (filter #(signed-datatype? (first %)))
        (map (fn [[src-dtype dst-dtype]]
               [[src-dtype dst-dtype]
                `(array->host-buffer-conversion ~src-dtype ~dst-dtype)]))
        (into {}))
   [:tvm-host-buffer :java-array]
   (->> full-conversion-sequence
        ;;Again, only jvm primitives for arrays
        (filter #(signed-datatype? (second %)))
        (map (fn [[src-dtype dst-dtype]]
               [[src-dtype dst-dtype]
                `(host-buffer->array-conversion ~src-dtype ~dst-dtype)]))
        (into {}))
   [:tvm-host-buffer :tvm-host-buffer]
   (->> full-conversion-sequence
        (map (fn [[src-dtype dst-dtype]]
               [[src-dtype dst-dtype]
                `(host-buffer->host-buffer-conversion ~src-dtype ~dst-dtype)]))
        (into {}))})


(def conversion-table (build-full-conversion))

(doseq [[types conversions] conversion-table]
  (marshal/add-copy-operation (first types) (second types) conversions))


(defn make-buffer-of-type
  [dtype elem-count-or-data]
  (if (signed-datatype? dtype)
    (->HostBuffer (jcpp-dtype/make-pointer-of-type dtype elem-count-or-data) dtype)
    (let [is-num? (number? elem-count-or-data)
          elem-count (long (if is-num?
                             elem-count-or-data
                             (m/ecount elem-count-or-data)))
          retval (->HostBuffer (jcpp-dtype/make-pointer-of-type (direct-conversion-map dtype)
                                                                elem-count)
                               dtype)]
      (when-not is-num?
        (dtype/copy-raw->item! elem-count-or-data retval 0))
      retval)))
