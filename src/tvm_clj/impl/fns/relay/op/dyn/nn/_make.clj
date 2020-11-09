(ns tvm-clj.impl.fns.relay.op.dyn.nn._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private pad-fnptr* (delay (base/name->global-function "relay.op.dyn.nn._make.pad")))
(defn pad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn.nn._make.pad"}
   (apply base/call-function @pad-fnptr* args)))

(defonce ^:private upsampling-fnptr* (delay (base/name->global-function "relay.op.dyn.nn._make.upsampling")))
(defn upsampling
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn.nn._make.upsampling"}
   (apply base/call-function @upsampling-fnptr* args)))

(defonce ^:private upsampling3d-fnptr* (delay (base/name->global-function "relay.op.dyn.nn._make.upsampling3d")))
(defn upsampling3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn.nn._make.upsampling3d"}
   (apply base/call-function @upsampling3d-fnptr* args)))

