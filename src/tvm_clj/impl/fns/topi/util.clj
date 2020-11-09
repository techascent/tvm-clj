(ns tvm-clj.impl.fns.topi.util
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private bilinear_sample_nchw-fnptr* (delay (base/name->global-function "topi.util.bilinear_sample_nchw")))
(defn bilinear_sample_nchw
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.util.bilinear_sample_nchw"}
   (apply base/call-function @bilinear_sample_nchw-fnptr* args)))

(defonce ^:private is_empty_shape-fnptr* (delay (base/name->global-function "topi.util.is_empty_shape")))
(defn is_empty_shape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.util.is_empty_shape"}
   (apply base/call-function @is_empty_shape-fnptr* args)))

