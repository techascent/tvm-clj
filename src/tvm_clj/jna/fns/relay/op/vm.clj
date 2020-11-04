(ns tvm-clj.jna.fns.relay.op.vm
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} invoke_tvm_op
(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.invoke_tvm_op"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reshape_tensor
(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.reshape_tensor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} shape_func
(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.shape_func"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} shape_of
(let [gfn* (delay (jna-base/name->global-function "relay.op.vm.shape_of"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

