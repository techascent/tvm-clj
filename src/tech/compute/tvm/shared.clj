(ns tech.compute.tvm.shared
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tech.datatype.base :as dtype]
            [tech.datatype.marshal :as marshal]
            [tech.datatype.core]))


(defprotocol PTVMDeviceInfo
  (device-type [item])
  (device-id [item]))


(def extended-primitive-type-map
  {:uint8 {:jvm-type :int8}
   :uint16 {:jvm-type :int16}
   :uint32 {:jvm-type :int32}
   :uint64 {:jvm-type :int64}})

(defn tvm-type->dtype-type
  [tvm-type]
  (get-in extended-primitive-type-map [tvm-type :jvm-type] tvm-type))


(defn enumerate-devices
  [^long device-type]
  (->> (range)
       (take-while #(tvm-core/device-exists? device-type %))))


(defmacro unsigned->signed
  [src-dtype val]
  (condp = src-dtype
    :uint8 `(bit-and (unchecked-short ~val) 0xFF)
    :uint16 `(bit-and (unchecked-int ~val) 0xFFFF)
    :uint32 `(bit-and (unchecked-long ~val) 0xFFFFFFFF)
    :uint64 `(bit-and (unchecked-long ~val) 0xFFFFFFFFFFFFFFFFF)
    `~val))


(defmacro ->unsigned
  [src-dtype val]
  (condp = src-dtype
    :uint8 `(unchecked-byte ~val)
    :uint16 `(unchecked-short ~val)
    :uint32 `(unchecked-int ~val)
    :uint64 `(unchecked-long ~val)
    `~val))


(defmacro cast-table-macro

  )


(def cast-table

  )




(defrecord HostBuffer [ptr tvm-dtype]
  dtype/PDatatype
  (get-datatype [this] tvm-dtype)

  dtype/PAccess
  marshal/PContainerType
  (container-type [this] :tvm-host-buffer))
