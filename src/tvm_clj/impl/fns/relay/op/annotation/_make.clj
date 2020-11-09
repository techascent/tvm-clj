(ns tvm-clj.jna.fns.relay.op.annotation._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.checkpoint"))]
  (defn checkpoint
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.annotation._make.checkpoint"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.compiler_begin"))]
  (defn compiler_begin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.annotation._make.compiler_begin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.compiler_end"))]
  (defn compiler_end
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.annotation._make.compiler_end"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.on_device"))]
  (defn on_device
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.annotation._make.on_device"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.annotation._make.stop_fusion"))]
  (defn stop_fusion
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.annotation._make.stop_fusion"}
     (apply jna-base/call-function @gfn* args))))

