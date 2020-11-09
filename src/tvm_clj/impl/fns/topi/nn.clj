(ns tvm-clj.jna.fns.topi.nn
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.adaptive_pool"))]
  (defn adaptive_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.adaptive_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.adaptive_pool3d"))]
  (defn adaptive_pool3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.adaptive_pool3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.batch_matmul"))]
  (defn batch_matmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.batch_matmul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.bias_add"))]
  (defn bias_add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.bias_add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.binarize_pack"))]
  (defn binarize_pack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.binarize_pack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.binary_dense"))]
  (defn binary_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.binary_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.dense"))]
  (defn dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.dilate"))]
  (defn dilate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.dilate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.flatten"))]
  (defn flatten
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.flatten"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.global_pool"))]
  (defn global_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.global_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.leaky_relu"))]
  (defn leaky_relu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.leaky_relu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.log_softmax"))]
  (defn log_softmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.log_softmax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.lrn"))]
  (defn lrn
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.lrn"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.pad"))]
  (defn pad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.pad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool"))]
  (defn pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool1d"))]
  (defn pool1d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.pool1d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool3d"))]
  (defn pool3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.pool3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool_grad"))]
  (defn pool_grad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.pool_grad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.prelu"))]
  (defn prelu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.prelu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.relu"))]
  (defn relu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.relu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.scale_shift_nchw"))]
  (defn scale_shift_nchw
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.scale_shift_nchw"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.scale_shift_nhwc"))]
  (defn scale_shift_nhwc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.scale_shift_nhwc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.nn.softmax"))]
  (defn softmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.nn.softmax"}
     (apply jna-base/call-function @gfn* args))))

