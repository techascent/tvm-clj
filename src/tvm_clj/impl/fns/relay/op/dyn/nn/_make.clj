(ns tvm-clj.jna.fns.relay.op.dyn.nn._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.nn._make.pad"))]
  (defn pad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn.nn._make.pad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.nn._make.upsampling"))]
  (defn upsampling
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn.nn._make.upsampling"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.nn._make.upsampling3d"))]
  (defn upsampling3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn.nn._make.upsampling3d"}
     (apply jna-base/call-function @gfn* args))))

