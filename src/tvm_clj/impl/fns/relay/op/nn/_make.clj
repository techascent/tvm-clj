(ns tvm-clj.jna.fns.relay.op.nn._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_avg_pool2d"))]
  (defn adaptive_avg_pool2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.adaptive_avg_pool2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_avg_pool3d"))]
  (defn adaptive_avg_pool3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.adaptive_avg_pool3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_max_pool2d"))]
  (defn adaptive_max_pool2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.adaptive_max_pool2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.adaptive_max_pool3d"))]
  (defn adaptive_max_pool3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.adaptive_max_pool3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool1d"))]
  (defn avg_pool1d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.avg_pool1d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool2d"))]
  (defn avg_pool2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.avg_pool2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool2d_grad"))]
  (defn avg_pool2d_grad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.avg_pool2d_grad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.avg_pool3d"))]
  (defn avg_pool3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.avg_pool3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.batch_flatten"))]
  (defn batch_flatten
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.batch_flatten"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.batch_matmul"))]
  (defn batch_matmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.batch_matmul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.batch_norm"))]
  (defn batch_norm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.batch_norm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bias_add"))]
  (defn bias_add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.bias_add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bitpack"))]
  (defn bitpack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.bitpack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bitserial_conv2d"))]
  (defn bitserial_conv2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.bitserial_conv2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.bitserial_dense"))]
  (defn bitserial_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.bitserial_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_NCHWc"))]
  (defn contrib_conv2d_NCHWc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv2d_NCHWc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_gemm_weight_transform"))]
  (defn contrib_conv2d_gemm_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv2d_gemm_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform"))]
  (defn contrib_conv2d_gemm_without_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform"))]
  (defn contrib_conv2d_winograd_nnpack_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_weight_transform"))]
  (defn contrib_conv2d_winograd_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv2d_winograd_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform"))]
  (defn contrib_conv2d_winograd_without_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv3d_winograd_weight_transform"))]
  (defn contrib_conv3d_winograd_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv3d_winograd_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform"))]
  (defn contrib_conv3d_winograd_without_weight_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.contrib_depthwise_conv2d_NCHWc"))]
  (defn contrib_depthwise_conv2d_NCHWc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.contrib_depthwise_conv2d_NCHWc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv1d"))]
  (defn conv1d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.conv1d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv1d_transpose"))]
  (defn conv1d_transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.conv1d_transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv2d"))]
  (defn conv2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.conv2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv2d_transpose"))]
  (defn conv2d_transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.conv2d_transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv3d"))]
  (defn conv3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.conv3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.conv3d_transpose"))]
  (defn conv3d_transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.conv3d_transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.correlation"))]
  (defn correlation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.correlation"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.cross_entropy"))]
  (defn cross_entropy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.cross_entropy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.cross_entropy_with_logits"))]
  (defn cross_entropy_with_logits
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.cross_entropy_with_logits"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.deformable_conv2d"))]
  (defn deformable_conv2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.deformable_conv2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.dense"))]
  (defn dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.depth_to_space"))]
  (defn depth_to_space
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.depth_to_space"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.dilate"))]
  (defn dilate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.dilate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.dropout"))]
  (defn dropout
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.dropout"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.fifo_buffer"))]
  (defn fifo_buffer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.fifo_buffer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.global_avg_pool2d"))]
  (defn global_avg_pool2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.global_avg_pool2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.global_max_pool2d"))]
  (defn global_max_pool2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.global_max_pool2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.group_norm"))]
  (defn group_norm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.group_norm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.instance_norm"))]
  (defn instance_norm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.instance_norm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.l2_normalize"))]
  (defn l2_normalize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.l2_normalize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.layer_norm"))]
  (defn layer_norm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.layer_norm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.leaky_relu"))]
  (defn leaky_relu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.leaky_relu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.log_softmax"))]
  (defn log_softmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.log_softmax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.lrn"))]
  (defn lrn
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.lrn"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool1d"))]
  (defn max_pool1d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.max_pool1d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool2d"))]
  (defn max_pool2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.max_pool2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool2d_grad"))]
  (defn max_pool2d_grad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.max_pool2d_grad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.max_pool3d"))]
  (defn max_pool3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.max_pool3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.mirror_pad"))]
  (defn mirror_pad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.mirror_pad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.pad"))]
  (defn pad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.pad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.prelu"))]
  (defn prelu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.prelu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.relu"))]
  (defn relu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.relu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.softmax"))]
  (defn softmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.softmax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.space_to_depth"))]
  (defn space_to_depth
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.space_to_depth"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.sparse_dense"))]
  (defn sparse_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.sparse_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.sparse_transpose"))]
  (defn sparse_transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.sparse_transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.upsampling"))]
  (defn upsampling
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.upsampling"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.nn._make.upsampling3d"))]
  (defn upsampling3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.nn._make.upsampling3d"}
     (apply jna-base/call-function @gfn* args))))

