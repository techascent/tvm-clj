(ns tvm-clj.jna.fns.relay.op.annotation._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} checkpoint
(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.checkpoint"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} compiler_begin
(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.compiler_begin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} compiler_end
(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.compiler_end"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} on_device
(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.on_device"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} stop_fusion
(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.stop_fusion"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

