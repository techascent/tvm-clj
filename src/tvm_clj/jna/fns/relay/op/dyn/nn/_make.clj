(ns tvm-clj.jna.fns.relay.op.dyn.nn._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pad
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.nn._make.pad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} upsampling
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.nn._make.upsampling"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} upsampling3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.nn._make.upsampling3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

