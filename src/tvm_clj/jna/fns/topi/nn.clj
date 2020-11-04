(ns tvm-clj.jna.fns.topi.nn
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adaptive_pool
(let [gfn* (delay (jna-base/name->global-function "topi.nn.adaptive_pool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adaptive_pool3d
(let [gfn* (delay (jna-base/name->global-function "topi.nn.adaptive_pool3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} batch_matmul
(let [gfn* (delay (jna-base/name->global-function "topi.nn.batch_matmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bias_add
(let [gfn* (delay (jna-base/name->global-function "topi.nn.bias_add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} binarize_pack
(let [gfn* (delay (jna-base/name->global-function "topi.nn.binarize_pack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} binary_dense
(let [gfn* (delay (jna-base/name->global-function "topi.nn.binary_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dense
(let [gfn* (delay (jna-base/name->global-function "topi.nn.dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dilate
(let [gfn* (delay (jna-base/name->global-function "topi.nn.dilate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} flatten
(let [gfn* (delay (jna-base/name->global-function "topi.nn.flatten"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} global_pool
(let [gfn* (delay (jna-base/name->global-function "topi.nn.global_pool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} leaky_relu
(let [gfn* (delay (jna-base/name->global-function "topi.nn.leaky_relu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log_softmax
(let [gfn* (delay (jna-base/name->global-function "topi.nn.log_softmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} lrn
(let [gfn* (delay (jna-base/name->global-function "topi.nn.lrn"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pad
(let [gfn* (delay (jna-base/name->global-function "topi.nn.pad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pool
(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pool1d
(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool1d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pool3d
(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pool_grad
(let [gfn* (delay (jna-base/name->global-function "topi.nn.pool_grad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} prelu
(let [gfn* (delay (jna-base/name->global-function "topi.nn.prelu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} relu
(let [gfn* (delay (jna-base/name->global-function "topi.nn.relu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} scale_shift_nchw
(let [gfn* (delay (jna-base/name->global-function "topi.nn.scale_shift_nchw"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} scale_shift_nhwc
(let [gfn* (delay (jna-base/name->global-function "topi.nn.scale_shift_nhwc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} softmax
(let [gfn* (delay (jna-base/name->global-function "topi.nn.softmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

