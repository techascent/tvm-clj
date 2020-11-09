(ns tvm-clj.impl.fns.relay.op.dyn._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private broadcast_to-fnptr* (delay (base/name->global-function "relay.op.dyn._make.broadcast_to")))
(defn broadcast_to
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.broadcast_to"}
   (apply base/call-function @broadcast_to-fnptr* args)))

(defonce ^:private full-fnptr* (delay (base/name->global-function "relay.op.dyn._make.full")))
(defn full
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.full"}
   (apply base/call-function @full-fnptr* args)))

(defonce ^:private one_hot-fnptr* (delay (base/name->global-function "relay.op.dyn._make.one_hot")))
(defn one_hot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.one_hot"}
   (apply base/call-function @one_hot-fnptr* args)))

(defonce ^:private ones-fnptr* (delay (base/name->global-function "relay.op.dyn._make.ones")))
(defn ones
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.ones"}
   (apply base/call-function @ones-fnptr* args)))

(defonce ^:private reshape-fnptr* (delay (base/name->global-function "relay.op.dyn._make.reshape")))
(defn reshape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.reshape"}
   (apply base/call-function @reshape-fnptr* args)))

(defonce ^:private strided_slice-fnptr* (delay (base/name->global-function "relay.op.dyn._make.strided_slice")))
(defn strided_slice
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.strided_slice"}
   (apply base/call-function @strided_slice-fnptr* args)))

(defonce ^:private tile-fnptr* (delay (base/name->global-function "relay.op.dyn._make.tile")))
(defn tile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.tile"}
   (apply base/call-function @tile-fnptr* args)))

(defonce ^:private topk-fnptr* (delay (base/name->global-function "relay.op.dyn._make.topk")))
(defn topk
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.topk"}
   (apply base/call-function @topk-fnptr* args)))

(defonce ^:private zeros-fnptr* (delay (base/name->global-function "relay.op.dyn._make.zeros")))
(defn zeros
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn._make.zeros"}
   (apply base/call-function @zeros-fnptr* args)))

