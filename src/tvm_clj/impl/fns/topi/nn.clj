(ns tvm-clj.impl.fns.topi.nn
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private adaptive_pool-fnptr* (delay (base/name->global-function "topi.nn.adaptive_pool")))
(defn adaptive_pool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.adaptive_pool"}
   (apply base/call-function @adaptive_pool-fnptr* args)))

(defonce ^:private adaptive_pool3d-fnptr* (delay (base/name->global-function "topi.nn.adaptive_pool3d")))
(defn adaptive_pool3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.adaptive_pool3d"}
   (apply base/call-function @adaptive_pool3d-fnptr* args)))

(defonce ^:private batch_matmul-fnptr* (delay (base/name->global-function "topi.nn.batch_matmul")))
(defn batch_matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.batch_matmul"}
   (apply base/call-function @batch_matmul-fnptr* args)))

(defonce ^:private bias_add-fnptr* (delay (base/name->global-function "topi.nn.bias_add")))
(defn bias_add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.bias_add"}
   (apply base/call-function @bias_add-fnptr* args)))

(defonce ^:private binarize_pack-fnptr* (delay (base/name->global-function "topi.nn.binarize_pack")))
(defn binarize_pack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.binarize_pack"}
   (apply base/call-function @binarize_pack-fnptr* args)))

(defonce ^:private binary_dense-fnptr* (delay (base/name->global-function "topi.nn.binary_dense")))
(defn binary_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.binary_dense"}
   (apply base/call-function @binary_dense-fnptr* args)))

(defonce ^:private dense-fnptr* (delay (base/name->global-function "topi.nn.dense")))
(defn dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.dense"}
   (apply base/call-function @dense-fnptr* args)))

(defonce ^:private dilate-fnptr* (delay (base/name->global-function "topi.nn.dilate")))
(defn dilate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.dilate"}
   (apply base/call-function @dilate-fnptr* args)))

(defonce ^:private flatten-fnptr* (delay (base/name->global-function "topi.nn.flatten")))
(defn flatten
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.flatten"}
   (apply base/call-function @flatten-fnptr* args)))

(defonce ^:private global_pool-fnptr* (delay (base/name->global-function "topi.nn.global_pool")))
(defn global_pool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.global_pool"}
   (apply base/call-function @global_pool-fnptr* args)))

(defonce ^:private leaky_relu-fnptr* (delay (base/name->global-function "topi.nn.leaky_relu")))
(defn leaky_relu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.leaky_relu"}
   (apply base/call-function @leaky_relu-fnptr* args)))

(defonce ^:private log_softmax-fnptr* (delay (base/name->global-function "topi.nn.log_softmax")))
(defn log_softmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.log_softmax"}
   (apply base/call-function @log_softmax-fnptr* args)))

(defonce ^:private lrn-fnptr* (delay (base/name->global-function "topi.nn.lrn")))
(defn lrn
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.lrn"}
   (apply base/call-function @lrn-fnptr* args)))

(defonce ^:private pad-fnptr* (delay (base/name->global-function "topi.nn.pad")))
(defn pad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.pad"}
   (apply base/call-function @pad-fnptr* args)))

(defonce ^:private pool-fnptr* (delay (base/name->global-function "topi.nn.pool")))
(defn pool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.pool"}
   (apply base/call-function @pool-fnptr* args)))

(defonce ^:private pool1d-fnptr* (delay (base/name->global-function "topi.nn.pool1d")))
(defn pool1d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.pool1d"}
   (apply base/call-function @pool1d-fnptr* args)))

(defonce ^:private pool3d-fnptr* (delay (base/name->global-function "topi.nn.pool3d")))
(defn pool3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.pool3d"}
   (apply base/call-function @pool3d-fnptr* args)))

(defonce ^:private pool_grad-fnptr* (delay (base/name->global-function "topi.nn.pool_grad")))
(defn pool_grad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.pool_grad"}
   (apply base/call-function @pool_grad-fnptr* args)))

(defonce ^:private prelu-fnptr* (delay (base/name->global-function "topi.nn.prelu")))
(defn prelu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.prelu"}
   (apply base/call-function @prelu-fnptr* args)))

(defonce ^:private relu-fnptr* (delay (base/name->global-function "topi.nn.relu")))
(defn relu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.relu"}
   (apply base/call-function @relu-fnptr* args)))

(defonce ^:private scale_shift_nchw-fnptr* (delay (base/name->global-function "topi.nn.scale_shift_nchw")))
(defn scale_shift_nchw
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.scale_shift_nchw"}
   (apply base/call-function @scale_shift_nchw-fnptr* args)))

(defonce ^:private scale_shift_nhwc-fnptr* (delay (base/name->global-function "topi.nn.scale_shift_nhwc")))
(defn scale_shift_nhwc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.scale_shift_nhwc"}
   (apply base/call-function @scale_shift_nhwc-fnptr* args)))

(defonce ^:private softmax-fnptr* (delay (base/name->global-function "topi.nn.softmax")))
(defn softmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.nn.softmax"}
   (apply base/call-function @softmax-fnptr* args)))

