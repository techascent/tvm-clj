(ns tvm-clj.impl.fns.relay.op.nn._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private adaptive_avg_pool2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.adaptive_avg_pool2d")))
(defn adaptive_avg_pool2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.adaptive_avg_pool2d"}
   (apply base/call-function @adaptive_avg_pool2d-fnptr* args)))

(defonce ^:private adaptive_avg_pool3d-fnptr* (delay (base/name->global-function "relay.op.nn._make.adaptive_avg_pool3d")))
(defn adaptive_avg_pool3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.adaptive_avg_pool3d"}
   (apply base/call-function @adaptive_avg_pool3d-fnptr* args)))

(defonce ^:private adaptive_max_pool2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.adaptive_max_pool2d")))
(defn adaptive_max_pool2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.adaptive_max_pool2d"}
   (apply base/call-function @adaptive_max_pool2d-fnptr* args)))

(defonce ^:private adaptive_max_pool3d-fnptr* (delay (base/name->global-function "relay.op.nn._make.adaptive_max_pool3d")))
(defn adaptive_max_pool3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.adaptive_max_pool3d"}
   (apply base/call-function @adaptive_max_pool3d-fnptr* args)))

(defonce ^:private avg_pool1d-fnptr* (delay (base/name->global-function "relay.op.nn._make.avg_pool1d")))
(defn avg_pool1d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.avg_pool1d"}
   (apply base/call-function @avg_pool1d-fnptr* args)))

(defonce ^:private avg_pool2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.avg_pool2d")))
(defn avg_pool2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.avg_pool2d"}
   (apply base/call-function @avg_pool2d-fnptr* args)))

(defonce ^:private avg_pool2d_grad-fnptr* (delay (base/name->global-function "relay.op.nn._make.avg_pool2d_grad")))
(defn avg_pool2d_grad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.avg_pool2d_grad"}
   (apply base/call-function @avg_pool2d_grad-fnptr* args)))

(defonce ^:private avg_pool3d-fnptr* (delay (base/name->global-function "relay.op.nn._make.avg_pool3d")))
(defn avg_pool3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.avg_pool3d"}
   (apply base/call-function @avg_pool3d-fnptr* args)))

(defonce ^:private batch_flatten-fnptr* (delay (base/name->global-function "relay.op.nn._make.batch_flatten")))
(defn batch_flatten
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.batch_flatten"}
   (apply base/call-function @batch_flatten-fnptr* args)))

(defonce ^:private batch_matmul-fnptr* (delay (base/name->global-function "relay.op.nn._make.batch_matmul")))
(defn batch_matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.batch_matmul"}
   (apply base/call-function @batch_matmul-fnptr* args)))

(defonce ^:private batch_norm-fnptr* (delay (base/name->global-function "relay.op.nn._make.batch_norm")))
(defn batch_norm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.batch_norm"}
   (apply base/call-function @batch_norm-fnptr* args)))

(defonce ^:private bias_add-fnptr* (delay (base/name->global-function "relay.op.nn._make.bias_add")))
(defn bias_add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.bias_add"}
   (apply base/call-function @bias_add-fnptr* args)))

(defonce ^:private bitpack-fnptr* (delay (base/name->global-function "relay.op.nn._make.bitpack")))
(defn bitpack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.bitpack"}
   (apply base/call-function @bitpack-fnptr* args)))

(defonce ^:private bitserial_conv2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.bitserial_conv2d")))
(defn bitserial_conv2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.bitserial_conv2d"}
   (apply base/call-function @bitserial_conv2d-fnptr* args)))

(defonce ^:private bitserial_dense-fnptr* (delay (base/name->global-function "relay.op.nn._make.bitserial_dense")))
(defn bitserial_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.bitserial_dense"}
   (apply base/call-function @bitserial_dense-fnptr* args)))

(defonce ^:private contrib_conv2d_NCHWc-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv2d_NCHWc")))
(defn contrib_conv2d_NCHWc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv2d_NCHWc"}
   (apply base/call-function @contrib_conv2d_NCHWc-fnptr* args)))

(defonce ^:private contrib_conv2d_gemm_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv2d_gemm_weight_transform")))
(defn contrib_conv2d_gemm_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv2d_gemm_weight_transform"}
   (apply base/call-function @contrib_conv2d_gemm_weight_transform-fnptr* args)))

(defonce ^:private contrib_conv2d_gemm_without_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform")))
(defn contrib_conv2d_gemm_without_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform"}
   (apply base/call-function @contrib_conv2d_gemm_without_weight_transform-fnptr* args)))

(defonce ^:private contrib_conv2d_winograd_nnpack_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform")))
(defn contrib_conv2d_winograd_nnpack_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform"}
   (apply base/call-function @contrib_conv2d_winograd_nnpack_weight_transform-fnptr* args)))

(defonce ^:private contrib_conv2d_winograd_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_weight_transform")))
(defn contrib_conv2d_winograd_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv2d_winograd_weight_transform"}
   (apply base/call-function @contrib_conv2d_winograd_weight_transform-fnptr* args)))

(defonce ^:private contrib_conv2d_winograd_without_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform")))
(defn contrib_conv2d_winograd_without_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform"}
   (apply base/call-function @contrib_conv2d_winograd_without_weight_transform-fnptr* args)))

(defonce ^:private contrib_conv3d_winograd_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv3d_winograd_weight_transform")))
(defn contrib_conv3d_winograd_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv3d_winograd_weight_transform"}
   (apply base/call-function @contrib_conv3d_winograd_weight_transform-fnptr* args)))

(defonce ^:private contrib_conv3d_winograd_without_weight_transform-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform")))
(defn contrib_conv3d_winograd_without_weight_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform"}
   (apply base/call-function @contrib_conv3d_winograd_without_weight_transform-fnptr* args)))

(defonce ^:private contrib_depthwise_conv2d_NCHWc-fnptr* (delay (base/name->global-function "relay.op.nn._make.contrib_depthwise_conv2d_NCHWc")))
(defn contrib_depthwise_conv2d_NCHWc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.contrib_depthwise_conv2d_NCHWc"}
   (apply base/call-function @contrib_depthwise_conv2d_NCHWc-fnptr* args)))

(defonce ^:private conv1d-fnptr* (delay (base/name->global-function "relay.op.nn._make.conv1d")))
(defn conv1d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.conv1d"}
   (apply base/call-function @conv1d-fnptr* args)))

(defonce ^:private conv1d_transpose-fnptr* (delay (base/name->global-function "relay.op.nn._make.conv1d_transpose")))
(defn conv1d_transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.conv1d_transpose"}
   (apply base/call-function @conv1d_transpose-fnptr* args)))

(defonce ^:private conv2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.conv2d")))
(defn conv2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.conv2d"}
   (apply base/call-function @conv2d-fnptr* args)))

(defonce ^:private conv2d_transpose-fnptr* (delay (base/name->global-function "relay.op.nn._make.conv2d_transpose")))
(defn conv2d_transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.conv2d_transpose"}
   (apply base/call-function @conv2d_transpose-fnptr* args)))

(defonce ^:private conv3d-fnptr* (delay (base/name->global-function "relay.op.nn._make.conv3d")))
(defn conv3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.conv3d"}
   (apply base/call-function @conv3d-fnptr* args)))

(defonce ^:private conv3d_transpose-fnptr* (delay (base/name->global-function "relay.op.nn._make.conv3d_transpose")))
(defn conv3d_transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.conv3d_transpose"}
   (apply base/call-function @conv3d_transpose-fnptr* args)))

(defonce ^:private correlation-fnptr* (delay (base/name->global-function "relay.op.nn._make.correlation")))
(defn correlation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.correlation"}
   (apply base/call-function @correlation-fnptr* args)))

(defonce ^:private cross_entropy-fnptr* (delay (base/name->global-function "relay.op.nn._make.cross_entropy")))
(defn cross_entropy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.cross_entropy"}
   (apply base/call-function @cross_entropy-fnptr* args)))

(defonce ^:private cross_entropy_with_logits-fnptr* (delay (base/name->global-function "relay.op.nn._make.cross_entropy_with_logits")))
(defn cross_entropy_with_logits
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.cross_entropy_with_logits"}
   (apply base/call-function @cross_entropy_with_logits-fnptr* args)))

(defonce ^:private deformable_conv2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.deformable_conv2d")))
(defn deformable_conv2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.deformable_conv2d"}
   (apply base/call-function @deformable_conv2d-fnptr* args)))

(defonce ^:private dense-fnptr* (delay (base/name->global-function "relay.op.nn._make.dense")))
(defn dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.dense"}
   (apply base/call-function @dense-fnptr* args)))

(defonce ^:private depth_to_space-fnptr* (delay (base/name->global-function "relay.op.nn._make.depth_to_space")))
(defn depth_to_space
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.depth_to_space"}
   (apply base/call-function @depth_to_space-fnptr* args)))

(defonce ^:private dilate-fnptr* (delay (base/name->global-function "relay.op.nn._make.dilate")))
(defn dilate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.dilate"}
   (apply base/call-function @dilate-fnptr* args)))

(defonce ^:private dropout-fnptr* (delay (base/name->global-function "relay.op.nn._make.dropout")))
(defn dropout
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.dropout"}
   (apply base/call-function @dropout-fnptr* args)))

(defonce ^:private fifo_buffer-fnptr* (delay (base/name->global-function "relay.op.nn._make.fifo_buffer")))
(defn fifo_buffer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.fifo_buffer"}
   (apply base/call-function @fifo_buffer-fnptr* args)))

(defonce ^:private global_avg_pool2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.global_avg_pool2d")))
(defn global_avg_pool2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.global_avg_pool2d"}
   (apply base/call-function @global_avg_pool2d-fnptr* args)))

(defonce ^:private global_max_pool2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.global_max_pool2d")))
(defn global_max_pool2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.global_max_pool2d"}
   (apply base/call-function @global_max_pool2d-fnptr* args)))

(defonce ^:private group_norm-fnptr* (delay (base/name->global-function "relay.op.nn._make.group_norm")))
(defn group_norm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.group_norm"}
   (apply base/call-function @group_norm-fnptr* args)))

(defonce ^:private instance_norm-fnptr* (delay (base/name->global-function "relay.op.nn._make.instance_norm")))
(defn instance_norm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.instance_norm"}
   (apply base/call-function @instance_norm-fnptr* args)))

(defonce ^:private l2_normalize-fnptr* (delay (base/name->global-function "relay.op.nn._make.l2_normalize")))
(defn l2_normalize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.l2_normalize"}
   (apply base/call-function @l2_normalize-fnptr* args)))

(defonce ^:private layer_norm-fnptr* (delay (base/name->global-function "relay.op.nn._make.layer_norm")))
(defn layer_norm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.layer_norm"}
   (apply base/call-function @layer_norm-fnptr* args)))

(defonce ^:private leaky_relu-fnptr* (delay (base/name->global-function "relay.op.nn._make.leaky_relu")))
(defn leaky_relu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.leaky_relu"}
   (apply base/call-function @leaky_relu-fnptr* args)))

(defonce ^:private log_softmax-fnptr* (delay (base/name->global-function "relay.op.nn._make.log_softmax")))
(defn log_softmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.log_softmax"}
   (apply base/call-function @log_softmax-fnptr* args)))

(defonce ^:private lrn-fnptr* (delay (base/name->global-function "relay.op.nn._make.lrn")))
(defn lrn
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.lrn"}
   (apply base/call-function @lrn-fnptr* args)))

(defonce ^:private max_pool1d-fnptr* (delay (base/name->global-function "relay.op.nn._make.max_pool1d")))
(defn max_pool1d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.max_pool1d"}
   (apply base/call-function @max_pool1d-fnptr* args)))

(defonce ^:private max_pool2d-fnptr* (delay (base/name->global-function "relay.op.nn._make.max_pool2d")))
(defn max_pool2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.max_pool2d"}
   (apply base/call-function @max_pool2d-fnptr* args)))

(defonce ^:private max_pool2d_grad-fnptr* (delay (base/name->global-function "relay.op.nn._make.max_pool2d_grad")))
(defn max_pool2d_grad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.max_pool2d_grad"}
   (apply base/call-function @max_pool2d_grad-fnptr* args)))

(defonce ^:private max_pool3d-fnptr* (delay (base/name->global-function "relay.op.nn._make.max_pool3d")))
(defn max_pool3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.max_pool3d"}
   (apply base/call-function @max_pool3d-fnptr* args)))

(defonce ^:private mirror_pad-fnptr* (delay (base/name->global-function "relay.op.nn._make.mirror_pad")))
(defn mirror_pad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.mirror_pad"}
   (apply base/call-function @mirror_pad-fnptr* args)))

(defonce ^:private pad-fnptr* (delay (base/name->global-function "relay.op.nn._make.pad")))
(defn pad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.pad"}
   (apply base/call-function @pad-fnptr* args)))

(defonce ^:private prelu-fnptr* (delay (base/name->global-function "relay.op.nn._make.prelu")))
(defn prelu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.prelu"}
   (apply base/call-function @prelu-fnptr* args)))

(defonce ^:private relu-fnptr* (delay (base/name->global-function "relay.op.nn._make.relu")))
(defn relu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.relu"}
   (apply base/call-function @relu-fnptr* args)))

(defonce ^:private softmax-fnptr* (delay (base/name->global-function "relay.op.nn._make.softmax")))
(defn softmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.softmax"}
   (apply base/call-function @softmax-fnptr* args)))

(defonce ^:private space_to_depth-fnptr* (delay (base/name->global-function "relay.op.nn._make.space_to_depth")))
(defn space_to_depth
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.space_to_depth"}
   (apply base/call-function @space_to_depth-fnptr* args)))

(defonce ^:private sparse_dense-fnptr* (delay (base/name->global-function "relay.op.nn._make.sparse_dense")))
(defn sparse_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.sparse_dense"}
   (apply base/call-function @sparse_dense-fnptr* args)))

(defonce ^:private sparse_transpose-fnptr* (delay (base/name->global-function "relay.op.nn._make.sparse_transpose")))
(defn sparse_transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.sparse_transpose"}
   (apply base/call-function @sparse_transpose-fnptr* args)))

(defonce ^:private upsampling-fnptr* (delay (base/name->global-function "relay.op.nn._make.upsampling")))
(defn upsampling
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.upsampling"}
   (apply base/call-function @upsampling-fnptr* args)))

(defonce ^:private upsampling3d-fnptr* (delay (base/name->global-function "relay.op.nn._make.upsampling3d")))
(defn upsampling3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.nn._make.upsampling3d"}
   (apply base/call-function @upsampling3d-fnptr* args)))

