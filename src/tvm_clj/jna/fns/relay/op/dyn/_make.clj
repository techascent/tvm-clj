(ns tvm-clj.jna.fns.relay.op.dyn._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} broadcast_to
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.broadcast_to"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} full
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.full"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} one_hot
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.one_hot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ones
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.ones"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reshape
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.reshape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} strided_slice
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.strided_slice"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tile
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.tile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} topk
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.topk"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} zeros
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn._make.zeros"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

