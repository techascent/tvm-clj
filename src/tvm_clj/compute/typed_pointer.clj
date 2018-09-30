(ns tvm-clj.compute.typed-pointer
  "Extensions to the typed pointer to integrate into the tech.compute ecosystem."
  (:require [tech.datatype.base :as dtype]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.compute.driver :as drv]
            [tvm-clj.base :as tvm-base]
            [tech.compute.tensor.utils :as tens-utils]
            [tech.typed-pointer :as typed-pointer])
  (:import [org.bytedeco.javacpp Pointer]
           [tech.typed_pointer TypedPointer]))


(defmethod tens-utils/dtype-cast :uint8
  [data dtype]
  (typed-pointer/unsigned->jvm :uint8 data))

(defmethod tens-utils/dtype-cast :uint16
  [data dtype]
  (typed-pointer/unsigned->jvm :uint16 data))

(defmethod tens-utils/dtype-cast :uint32
  [data dtype]
  (typed-pointer/unsigned->jvm :uint32 data))

(defmethod tens-utils/dtype-cast :uint64
  [data dtype]
  (typed-pointer/unsigned->jvm :uint64 data))


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


(extend-type TypedPointer
  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (typed-pointer/->TypedPointer (jcpp-pointer-sub-buffer (typed-pointer/->ptr buffer) offset length)
                                  (dtype/get-datatype buffer)))
  (alias? [lhs rhs]
    (jcpp-pointer-alias? (typed-pointer/->ptr lhs)
                         (typed-pointer/->ptr rhs)))
  (partially-alias? [lhs rhs]
    (jcpp-pointer-partial-alias? (typed-pointer/->ptr lhs)
                                 (typed-pointer/->ptr rhs)))
  tvm-base/PToTVM
  (->tvm [item] (typed-pointer/->ptr item)))
