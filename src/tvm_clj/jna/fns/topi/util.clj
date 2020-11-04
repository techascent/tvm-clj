(ns tvm-clj.jna.fns.topi.util
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.util.bilinear_sample_nchw"))]
  (defn bilinear_sample_nchw
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.util.bilinear_sample_nchw"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.util.is_empty_shape"))]
  (defn is_empty_shape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.util.is_empty_shape"}
     (apply jna-base/call-function @gfn* args))))

