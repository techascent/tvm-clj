(ns tvm-clj.impl.fns.relay.op.memory._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private FlattenTupleType-fnptr* (delay (base/name->global-function "relay.op.memory._make.FlattenTupleType")))
(defn FlattenTupleType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.memory._make.FlattenTupleType"}
   (apply base/call-function @FlattenTupleType-fnptr* args)))

(defonce ^:private FromTupleType-fnptr* (delay (base/name->global-function "relay.op.memory._make.FromTupleType")))
(defn FromTupleType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.memory._make.FromTupleType"}
   (apply base/call-function @FromTupleType-fnptr* args)))

(defonce ^:private ToTupleType-fnptr* (delay (base/name->global-function "relay.op.memory._make.ToTupleType")))
(defn ToTupleType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.memory._make.ToTupleType"}
   (apply base/call-function @ToTupleType-fnptr* args)))

(defonce ^:private alloc_storage-fnptr* (delay (base/name->global-function "relay.op.memory._make.alloc_storage")))
(defn alloc_storage
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.memory._make.alloc_storage"}
   (apply base/call-function @alloc_storage-fnptr* args)))

(defonce ^:private alloc_tensor-fnptr* (delay (base/name->global-function "relay.op.memory._make.alloc_tensor")))
(defn alloc_tensor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.memory._make.alloc_tensor"}
   (apply base/call-function @alloc_tensor-fnptr* args)))

