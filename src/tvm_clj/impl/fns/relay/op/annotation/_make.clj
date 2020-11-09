(ns tvm-clj.impl.fns.relay.op.annotation._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private checkpoint-fnptr* (delay (base/name->global-function "relay.op.annotation._make.checkpoint")))
(defn checkpoint
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.annotation._make.checkpoint"}
   (apply base/call-function @checkpoint-fnptr* args)))

(defonce ^:private compiler_begin-fnptr* (delay (base/name->global-function "relay.op.annotation._make.compiler_begin")))
(defn compiler_begin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.annotation._make.compiler_begin"}
   (apply base/call-function @compiler_begin-fnptr* args)))

(defonce ^:private compiler_end-fnptr* (delay (base/name->global-function "relay.op.annotation._make.compiler_end")))
(defn compiler_end
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.annotation._make.compiler_end"}
   (apply base/call-function @compiler_end-fnptr* args)))

(defonce ^:private on_device-fnptr* (delay (base/name->global-function "relay.op.annotation._make.on_device")))
(defn on_device
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.annotation._make.on_device"}
   (apply base/call-function @on_device-fnptr* args)))

(defonce ^:private stop_fusion-fnptr* (delay (base/name->global-function "relay.op.annotation._make.stop_fusion")))
(defn stop_fusion
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.annotation._make.stop_fusion"}
   (apply base/call-function @stop_fusion-fnptr* args)))

