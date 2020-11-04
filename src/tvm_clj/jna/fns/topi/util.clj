(ns tvm-clj.jna.fns.topi.util
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bilinear_sample_nchw
(let [gfn* (delay (jna-base/name->global-function "topi.util.bilinear_sample_nchw"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} is_empty_shape
(let [gfn* (delay (jna-base/name->global-function "topi.util.is_empty_shape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

