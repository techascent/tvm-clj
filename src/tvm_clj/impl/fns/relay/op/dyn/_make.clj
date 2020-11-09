(ns tvm-clj.jna.fns.relay.op.dyn._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.broadcast_to"))]
  (defn broadcast_to
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.broadcast_to"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.full"))]
  (defn full
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.full"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.one_hot"))]
  (defn one_hot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.one_hot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.ones"))]
  (defn ones
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.ones"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.reshape"))]
  (defn reshape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.reshape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.strided_slice"))]
  (defn strided_slice
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.strided_slice"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.tile"))]
  (defn tile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.tile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.topk"))]
  (defn topk
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.topk"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.zeros"))]
  (defn zeros
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn._make.zeros"}
     (apply jna-base/call-function @gfn* args))))

