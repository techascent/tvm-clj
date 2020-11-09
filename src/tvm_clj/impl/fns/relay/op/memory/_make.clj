(ns tvm-clj.jna.fns.relay.op.memory._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.FlattenTupleType"))]
  (defn FlattenTupleType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.memory._make.FlattenTupleType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.FromTupleType"))]
  (defn FromTupleType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.memory._make.FromTupleType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.ToTupleType"))]
  (defn ToTupleType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.memory._make.ToTupleType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.alloc_storage"))]
  (defn alloc_storage
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.memory._make.alloc_storage"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.alloc_tensor"))]
  (defn alloc_tensor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.memory._make.alloc_tensor"}
     (apply jna-base/call-function @gfn* args))))

