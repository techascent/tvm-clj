(ns tvm-clj.jna.fns.relay.op.vm
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.invoke_tvm_op"))]
  (defn invoke_tvm_op
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vm.invoke_tvm_op"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.reshape_tensor"))]
  (defn reshape_tensor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vm.reshape_tensor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.shape_func"))]
  (defn shape_func
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vm.shape_func"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.shape_of"))]
  (defn shape_of
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vm.shape_of"}
     (apply jna-base/call-function @gfn* args))))

