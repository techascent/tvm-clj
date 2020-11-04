(ns tvm-clj.jna.fns.relay.op.image._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} affine_grid
(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.affine_grid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} crop_and_resize
(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.crop_and_resize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dilation2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.dilation2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} grid_sample
(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.grid_sample"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} resize
(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.resize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} resize3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.image._make.resize3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

