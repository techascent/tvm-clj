(ns tvm-clj.bindings.protocols
  (:require [tech.v3.jna :as jna])
  (:import [com.sun.jna Pointer]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defprotocol PJVMTypeToTVMValue
  "Convert something to a [long tvm-value-type] pair"
  (->tvm-value [jvm-type]))


(defprotocol PToTVM
  "Convert something to some level of tvm type."
  (->tvm [item]))


(defprotocol PConvertToNode
  (->node [item]))


(defprotocol PTVMNode
  (is-node-handle? [item])
  (node-type-index [item])
  (node-type-name [item]))


(extend-type Object
  PTVMNode
  (is-node-handle? [item] false))


(defprotocol PTVMDeviceId
  (device-id [item]))


(defprotocol PTVMDeviceType
  (device-type [item]))


(extend-type Object
  PTVMDeviceId
  (device-id [item] 0)
  PTVMDeviceType
  (device-type [item] :cpu))


(defprotocol PByteOffset
  "Some buffers you cant offset (opengl, for instance).
So buffers have a logical byte-offset that is passed to functions.
So we need to get the actual base ptr sometimes."
  (byte-offset [item])
  (base-ptr [item]))


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
  (->tvm-value [value] [(Pointer/nativeValue (jna/string->ptr value {:track-type :auto}))
                        :string])

  nil
  (->tvm-value [value]
    [(long 0) :null]))
