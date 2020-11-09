(ns tvm-clj.impl.fns.relay.op.vm
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private invoke_tvm_op-fnptr* (delay (base/name->global-function "relay.op.vm.invoke_tvm_op")))
(defn invoke_tvm_op
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vm.invoke_tvm_op"}
   (apply base/call-function @invoke_tvm_op-fnptr* args)))

(defonce ^:private reshape_tensor-fnptr* (delay (base/name->global-function "relay.op.vm.reshape_tensor")))
(defn reshape_tensor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vm.reshape_tensor"}
   (apply base/call-function @reshape_tensor-fnptr* args)))

(defonce ^:private shape_func-fnptr* (delay (base/name->global-function "relay.op.vm.shape_func")))
(defn shape_func
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vm.shape_func"}
   (apply base/call-function @shape_func-fnptr* args)))

(defonce ^:private shape_of-fnptr* (delay (base/name->global-function "relay.op.vm.shape_of")))
(defn shape_of
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vm.shape_of"}
   (apply base/call-function @shape_of-fnptr* args)))

