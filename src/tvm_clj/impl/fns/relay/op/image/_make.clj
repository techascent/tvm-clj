(ns tvm-clj.jna.fns.relay.op.image._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.affine_grid"))]
  (defn affine_grid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.image._make.affine_grid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.crop_and_resize"))]
  (defn crop_and_resize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.image._make.crop_and_resize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.dilation2d"))]
  (defn dilation2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.image._make.dilation2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.grid_sample"))]
  (defn grid_sample
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.image._make.grid_sample"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.resize"))]
  (defn resize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.image._make.resize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.resize3d"))]
  (defn resize3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.image._make.resize3d"}
     (apply jna-base/call-function @gfn* args))))

