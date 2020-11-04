(ns tvm-clj.jna.fns.relay.op.nn._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adaptive_avg_pool2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_avg_pool2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adaptive_avg_pool3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_avg_pool3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adaptive_max_pool2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_max_pool2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adaptive_max_pool3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_max_pool3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} avg_pool1d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool1d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} avg_pool2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} avg_pool2d_grad
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool2d_grad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} avg_pool3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} batch_flatten
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.batch_flatten"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} batch_matmul
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.batch_matmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} batch_norm
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.batch_norm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bias_add
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bias_add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitpack
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bitpack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitserial_conv2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bitserial_conv2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitserial_dense
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bitserial_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv2d_NCHWc
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_NCHWc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv2d_gemm_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_gemm_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv2d_gemm_without_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv2d_winograd_nnpack_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv2d_winograd_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv2d_winograd_without_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv3d_winograd_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv3d_winograd_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_conv3d_winograd_without_weight_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_depthwise_conv2d_NCHWc
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_depthwise_conv2d_NCHWc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv1d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv1d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv1d_transpose
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv1d_transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv2d_transpose
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv2d_transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv3d_transpose
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv3d_transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} correlation
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.correlation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cross_entropy
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.cross_entropy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cross_entropy_with_logits
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.cross_entropy_with_logits"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} deformable_conv2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.deformable_conv2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dense
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} depth_to_space
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.depth_to_space"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dilate
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.dilate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dropout
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.dropout"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fifo_buffer
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.fifo_buffer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} global_avg_pool2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.global_avg_pool2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} global_max_pool2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.global_max_pool2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} group_norm
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.group_norm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} instance_norm
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.instance_norm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} l2_normalize
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.l2_normalize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} layer_norm
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.layer_norm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} leaky_relu
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.leaky_relu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log_softmax
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.log_softmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} lrn
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.lrn"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max_pool1d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool1d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max_pool2d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max_pool2d_grad
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool2d_grad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max_pool3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} mirror_pad
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.mirror_pad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pad
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.pad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} prelu
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.prelu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} relu
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.relu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} softmax
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.softmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} space_to_depth
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.space_to_depth"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sparse_dense
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.sparse_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sparse_transpose
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.sparse_transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} upsampling
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.upsampling"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} upsampling3d
(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.upsampling3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

