(ns tvm-clj.impl.fns.relay.op.image._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private affine_grid-fnptr* (delay (base/name->global-function "relay.op.image._make.affine_grid")))
(defn affine_grid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.image._make.affine_grid"}
   (apply base/call-function @affine_grid-fnptr* args)))

(defonce ^:private crop_and_resize-fnptr* (delay (base/name->global-function "relay.op.image._make.crop_and_resize")))
(defn crop_and_resize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.image._make.crop_and_resize"}
   (apply base/call-function @crop_and_resize-fnptr* args)))

(defonce ^:private dilation2d-fnptr* (delay (base/name->global-function "relay.op.image._make.dilation2d")))
(defn dilation2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.image._make.dilation2d"}
   (apply base/call-function @dilation2d-fnptr* args)))

(defonce ^:private grid_sample-fnptr* (delay (base/name->global-function "relay.op.image._make.grid_sample")))
(defn grid_sample
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.image._make.grid_sample"}
   (apply base/call-function @grid_sample-fnptr* args)))

(defonce ^:private resize-fnptr* (delay (base/name->global-function "relay.op.image._make.resize")))
(defn resize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.image._make.resize"}
   (apply base/call-function @resize-fnptr* args)))

(defonce ^:private resize3d-fnptr* (delay (base/name->global-function "relay.op.image._make.resize3d")))
(defn resize3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.image._make.resize3d"}
   (apply base/call-function @resize3d-fnptr* args)))

