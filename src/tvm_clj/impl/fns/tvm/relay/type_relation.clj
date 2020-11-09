(ns tvm-clj.impl.fns.tvm.relay.type_relation
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AdaptiveAvgPool2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AdaptiveAvgPool2D")))
(defn AdaptiveAvgPool2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AdaptiveAvgPool2D"}
   (apply base/call-function @AdaptiveAvgPool2D-fnptr* args)))

(defonce ^:private AdaptiveAvgPool3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AdaptiveAvgPool3D")))
(defn AdaptiveAvgPool3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AdaptiveAvgPool3D"}
   (apply base/call-function @AdaptiveAvgPool3D-fnptr* args)))

(defonce ^:private AdaptiveMaxPool2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AdaptiveMaxPool2D")))
(defn AdaptiveMaxPool2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AdaptiveMaxPool2D"}
   (apply base/call-function @AdaptiveMaxPool2D-fnptr* args)))

(defonce ^:private AdaptiveMaxPool3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AdaptiveMaxPool3D")))
(defn AdaptiveMaxPool3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AdaptiveMaxPool3D"}
   (apply base/call-function @AdaptiveMaxPool3D-fnptr* args)))

(defonce ^:private AdvIndex-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AdvIndex")))
(defn AdvIndex
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AdvIndex"}
   (apply base/call-function @AdvIndex-fnptr* args)))

(defonce ^:private AffineGrid-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AffineGrid")))
(defn AffineGrid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AffineGrid"}
   (apply base/call-function @AffineGrid-fnptr* args)))

(defonce ^:private AllocStorage-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AllocStorage")))
(defn AllocStorage
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AllocStorage"}
   (apply base/call-function @AllocStorage-fnptr* args)))

(defonce ^:private AllocTensor-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AllocTensor")))
(defn AllocTensor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AllocTensor"}
   (apply base/call-function @AllocTensor-fnptr* args)))

(defonce ^:private Arange-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Arange")))
(defn Arange
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Arange"}
   (apply base/call-function @Arange-fnptr* args)))

(defonce ^:private ArgReduce-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ArgReduce")))
(defn ArgReduce
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ArgReduce"}
   (apply base/call-function @ArgReduce-fnptr* args)))

(defonce ^:private ArgWhere-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ArgWhere")))
(defn ArgWhere
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ArgWhere"}
   (apply base/call-function @ArgWhere-fnptr* args)))

(defonce ^:private Argsort-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Argsort")))
(defn Argsort
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Argsort"}
   (apply base/call-function @Argsort-fnptr* args)))

(defonce ^:private AvgPool1D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AvgPool1D")))
(defn AvgPool1D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AvgPool1D"}
   (apply base/call-function @AvgPool1D-fnptr* args)))

(defonce ^:private AvgPool2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AvgPool2D")))
(defn AvgPool2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AvgPool2D"}
   (apply base/call-function @AvgPool2D-fnptr* args)))

(defonce ^:private AvgPool3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.AvgPool3D")))
(defn AvgPool3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.AvgPool3D"}
   (apply base/call-function @AvgPool3D-fnptr* args)))

(defonce ^:private BatchFlatten-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BatchFlatten")))
(defn BatchFlatten
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BatchFlatten"}
   (apply base/call-function @BatchFlatten-fnptr* args)))

(defonce ^:private BatchMatmul-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BatchMatmul")))
(defn BatchMatmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BatchMatmul"}
   (apply base/call-function @BatchMatmul-fnptr* args)))

(defonce ^:private BatchNorm-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BatchNorm")))
(defn BatchNorm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BatchNorm"}
   (apply base/call-function @BatchNorm-fnptr* args)))

(defonce ^:private BiasAdd-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BiasAdd")))
(defn BiasAdd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BiasAdd"}
   (apply base/call-function @BiasAdd-fnptr* args)))

(defonce ^:private BinaryConv2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BinaryConv2D")))
(defn BinaryConv2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BinaryConv2D"}
   (apply base/call-function @BinaryConv2D-fnptr* args)))

(defonce ^:private BinaryDense-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BinaryDense")))
(defn BinaryDense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BinaryDense"}
   (apply base/call-function @BinaryDense-fnptr* args)))

(defonce ^:private BitPack-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BitPack")))
(defn BitPack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BitPack"}
   (apply base/call-function @BitPack-fnptr* args)))

(defonce ^:private BroadCastTo-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BroadCastTo")))
(defn BroadCastTo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BroadCastTo"}
   (apply base/call-function @BroadCastTo-fnptr* args)))

(defonce ^:private BroadCastToLike-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BroadCastToLike")))
(defn BroadCastToLike
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BroadCastToLike"}
   (apply base/call-function @BroadCastToLike-fnptr* args)))

(defonce ^:private Broadcast-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Broadcast")))
(defn Broadcast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Broadcast"}
   (apply base/call-function @Broadcast-fnptr* args)))

(defonce ^:private BroadcastComp-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.BroadcastComp")))
(defn BroadcastComp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.BroadcastComp"}
   (apply base/call-function @BroadcastComp-fnptr* args)))

(defonce ^:private Cast-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Cast")))
(defn Cast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Cast"}
   (apply base/call-function @Cast-fnptr* args)))

(defonce ^:private CastLike-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.CastLike")))
(defn CastLike
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.CastLike"}
   (apply base/call-function @CastLike-fnptr* args)))

(defonce ^:private CollapseSumLike-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.CollapseSumLike")))
(defn CollapseSumLike
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.CollapseSumLike"}
   (apply base/call-function @CollapseSumLike-fnptr* args)))

(defonce ^:private CollapseSumTo-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.CollapseSumTo")))
(defn CollapseSumTo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.CollapseSumTo"}
   (apply base/call-function @CollapseSumTo-fnptr* args)))

(defonce ^:private Concatenate-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Concatenate")))
(defn Concatenate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Concatenate"}
   (apply base/call-function @Concatenate-fnptr* args)))

(defonce ^:private Conv1D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv1D")))
(defn Conv1D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv1D"}
   (apply base/call-function @Conv1D-fnptr* args)))

(defonce ^:private Conv1DTranspose-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv1DTranspose")))
(defn Conv1DTranspose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv1DTranspose"}
   (apply base/call-function @Conv1DTranspose-fnptr* args)))

(defonce ^:private Conv2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2D")))
(defn Conv2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2D"}
   (apply base/call-function @Conv2D-fnptr* args)))

(defonce ^:private Conv2DGemm-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DGemm")))
(defn Conv2DGemm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DGemm"}
   (apply base/call-function @Conv2DGemm-fnptr* args)))

(defonce ^:private Conv2DGemmWeightTransform-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DGemmWeightTransform")))
(defn Conv2DGemmWeightTransform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DGemmWeightTransform"}
   (apply base/call-function @Conv2DGemmWeightTransform-fnptr* args)))

(defonce ^:private Conv2DNCHWc-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DNCHWc")))
(defn Conv2DNCHWc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DNCHWc"}
   (apply base/call-function @Conv2DNCHWc-fnptr* args)))

(defonce ^:private Conv2DTranspose-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DTranspose")))
(defn Conv2DTranspose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DTranspose"}
   (apply base/call-function @Conv2DTranspose-fnptr* args)))

(defonce ^:private Conv2DWinograd-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DWinograd")))
(defn Conv2DWinograd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DWinograd"}
   (apply base/call-function @Conv2DWinograd-fnptr* args)))

(defonce ^:private Conv2DWinogradNNPACKWeightTransform-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DWinogradNNPACKWeightTransform")))
(defn Conv2DWinogradNNPACKWeightTransform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DWinogradNNPACKWeightTransform"}
   (apply base/call-function @Conv2DWinogradNNPACKWeightTransform-fnptr* args)))

(defonce ^:private Conv2DWinogradWeightTransform-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv2DWinogradWeightTransform")))
(defn Conv2DWinogradWeightTransform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv2DWinogradWeightTransform"}
   (apply base/call-function @Conv2DWinogradWeightTransform-fnptr* args)))

(defonce ^:private Conv3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv3D")))
(defn Conv3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv3D"}
   (apply base/call-function @Conv3D-fnptr* args)))

(defonce ^:private Conv3DTranspose-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv3DTranspose")))
(defn Conv3DTranspose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv3DTranspose"}
   (apply base/call-function @Conv3DTranspose-fnptr* args)))

(defonce ^:private Conv3DWinograd-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv3DWinograd")))
(defn Conv3DWinograd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv3DWinograd"}
   (apply base/call-function @Conv3DWinograd-fnptr* args)))

(defonce ^:private Conv3DWinogradWeightTransform-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Conv3DWinogradWeightTransform")))
(defn Conv3DWinogradWeightTransform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Conv3DWinogradWeightTransform"}
   (apply base/call-function @Conv3DWinogradWeightTransform-fnptr* args)))

(defonce ^:private Correlation-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Correlation")))
(defn Correlation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Correlation"}
   (apply base/call-function @Correlation-fnptr* args)))

(defonce ^:private CropAndResize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.CropAndResize")))
(defn CropAndResize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.CropAndResize"}
   (apply base/call-function @CropAndResize-fnptr* args)))

(defonce ^:private CrossEntropy-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.CrossEntropy")))
(defn CrossEntropy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.CrossEntropy"}
   (apply base/call-function @CrossEntropy-fnptr* args)))

(defonce ^:private Debug-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Debug")))
(defn Debug
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Debug"}
   (apply base/call-function @Debug-fnptr* args)))

(defonce ^:private DeformableConv2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DeformableConv2D")))
(defn DeformableConv2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DeformableConv2D"}
   (apply base/call-function @DeformableConv2D-fnptr* args)))

(defonce ^:private Dense-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Dense")))
(defn Dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Dense"}
   (apply base/call-function @Dense-fnptr* args)))

(defonce ^:private DepthToSpace-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DepthToSpace")))
(defn DepthToSpace
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DepthToSpace"}
   (apply base/call-function @DepthToSpace-fnptr* args)))

(defonce ^:private Dequantize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Dequantize")))
(defn Dequantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Dequantize"}
   (apply base/call-function @Dequantize-fnptr* args)))

(defonce ^:private Dilate-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Dilate")))
(defn Dilate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Dilate"}
   (apply base/call-function @Dilate-fnptr* args)))

(defonce ^:private Dilation2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Dilation2D")))
(defn Dilation2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Dilation2D"}
   (apply base/call-function @Dilation2D-fnptr* args)))

(defonce ^:private Dropout-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Dropout")))
(defn Dropout
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Dropout"}
   (apply base/call-function @Dropout-fnptr* args)))

(defonce ^:private DynOneHot-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynOneHot")))
(defn DynOneHot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynOneHot"}
   (apply base/call-function @DynOneHot-fnptr* args)))

(defonce ^:private DynResize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynResize")))
(defn DynResize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynResize"}
   (apply base/call-function @DynResize-fnptr* args)))

(defonce ^:private DynStridedSlice-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynStridedSlice")))
(defn DynStridedSlice
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynStridedSlice"}
   (apply base/call-function @DynStridedSlice-fnptr* args)))

(defonce ^:private DynTopK-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynTopK")))
(defn DynTopK
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynTopK"}
   (apply base/call-function @DynTopK-fnptr* args)))

(defonce ^:private DynamicBroadCastTo-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicBroadCastTo")))
(defn DynamicBroadCastTo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicBroadCastTo"}
   (apply base/call-function @DynamicBroadCastTo-fnptr* args)))

(defonce ^:private DynamicFull-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicFull")))
(defn DynamicFull
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicFull"}
   (apply base/call-function @DynamicFull-fnptr* args)))

(defonce ^:private DynamicInitOp-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicInitOp")))
(defn DynamicInitOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicInitOp"}
   (apply base/call-function @DynamicInitOp-fnptr* args)))

(defonce ^:private DynamicPad-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicPad")))
(defn DynamicPad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicPad"}
   (apply base/call-function @DynamicPad-fnptr* args)))

(defonce ^:private DynamicReshape-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicReshape")))
(defn DynamicReshape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicReshape"}
   (apply base/call-function @DynamicReshape-fnptr* args)))

(defonce ^:private DynamicTile-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicTile")))
(defn DynamicTile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicTile"}
   (apply base/call-function @DynamicTile-fnptr* args)))

(defonce ^:private DynamicUpSampling-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicUpSampling")))
(defn DynamicUpSampling
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicUpSampling"}
   (apply base/call-function @DynamicUpSampling-fnptr* args)))

(defonce ^:private DynamicUpSampling3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.DynamicUpSampling3D")))
(defn DynamicUpSampling3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.DynamicUpSampling3D"}
   (apply base/call-function @DynamicUpSampling3D-fnptr* args)))

(defonce ^:private ExpandDims-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ExpandDims")))
(defn ExpandDims
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ExpandDims"}
   (apply base/call-function @ExpandDims-fnptr* args)))

(defonce ^:private FIFOBuffer-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.FIFOBuffer")))
(defn FIFOBuffer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.FIFOBuffer"}
   (apply base/call-function @FIFOBuffer-fnptr* args)))

(defonce ^:private Full-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Full")))
(defn Full
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Full"}
   (apply base/call-function @Full-fnptr* args)))

(defonce ^:private FullLike-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.FullLike")))
(defn FullLike
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.FullLike"}
   (apply base/call-function @FullLike-fnptr* args)))

(defonce ^:private Gather-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Gather")))
(defn Gather
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Gather"}
   (apply base/call-function @Gather-fnptr* args)))

(defonce ^:private GatherND-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.GatherND")))
(defn GatherND
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.GatherND"}
   (apply base/call-function @GatherND-fnptr* args)))

(defonce ^:private GetValidCount-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.GetValidCount")))
(defn GetValidCount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.GetValidCount"}
   (apply base/call-function @GetValidCount-fnptr* args)))

(defonce ^:private GlobalAvgPool2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.GlobalAvgPool2D")))
(defn GlobalAvgPool2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.GlobalAvgPool2D"}
   (apply base/call-function @GlobalAvgPool2D-fnptr* args)))

(defonce ^:private GlobalMaxPool2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.GlobalMaxPool2D")))
(defn GlobalMaxPool2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.GlobalMaxPool2D"}
   (apply base/call-function @GlobalMaxPool2D-fnptr* args)))

(defonce ^:private GridSample-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.GridSample")))
(defn GridSample
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.GridSample"}
   (apply base/call-function @GridSample-fnptr* args)))

(defonce ^:private GroupNorm-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.GroupNorm")))
(defn GroupNorm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.GroupNorm"}
   (apply base/call-function @GroupNorm-fnptr* args)))

(defonce ^:private Identity-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Identity")))
(defn Identity
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Identity"}
   (apply base/call-function @Identity-fnptr* args)))

(defonce ^:private IdentityCompRel-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.IdentityCompRel")))
(defn IdentityCompRel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.IdentityCompRel"}
   (apply base/call-function @IdentityCompRel-fnptr* args)))

(defonce ^:private InitOp-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.InitOp")))
(defn InitOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.InitOp"}
   (apply base/call-function @InitOp-fnptr* args)))

(defonce ^:private InstanceNorm-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.InstanceNorm")))
(defn InstanceNorm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.InstanceNorm"}
   (apply base/call-function @InstanceNorm-fnptr* args)))

(defonce ^:private InvokeTVMOp-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.InvokeTVMOp")))
(defn InvokeTVMOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.InvokeTVMOp"}
   (apply base/call-function @InvokeTVMOp-fnptr* args)))

(defonce ^:private Kill-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Kill")))
(defn Kill
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Kill"}
   (apply base/call-function @Kill-fnptr* args)))

(defonce ^:private LayerNorm-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.LayerNorm")))
(defn LayerNorm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.LayerNorm"}
   (apply base/call-function @LayerNorm-fnptr* args)))

(defonce ^:private MatrixSetDiag-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MatrixSetDiag")))
(defn MatrixSetDiag
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MatrixSetDiag"}
   (apply base/call-function @MatrixSetDiag-fnptr* args)))

(defonce ^:private MaxPool1D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MaxPool1D")))
(defn MaxPool1D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MaxPool1D"}
   (apply base/call-function @MaxPool1D-fnptr* args)))

(defonce ^:private MaxPool2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MaxPool2D")))
(defn MaxPool2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MaxPool2D"}
   (apply base/call-function @MaxPool2D-fnptr* args)))

(defonce ^:private MaxPool2DGrad-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MaxPool2DGrad")))
(defn MaxPool2DGrad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MaxPool2DGrad"}
   (apply base/call-function @MaxPool2DGrad-fnptr* args)))

(defonce ^:private MaxPool3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MaxPool3D")))
(defn MaxPool3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MaxPool3D"}
   (apply base/call-function @MaxPool3D-fnptr* args)))

(defonce ^:private Meshgrid-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Meshgrid")))
(defn Meshgrid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Meshgrid"}
   (apply base/call-function @Meshgrid-fnptr* args)))

(defonce ^:private MetaRef-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MetaRef")))
(defn MetaRef
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MetaRef"}
   (apply base/call-function @MetaRef-fnptr* args)))

(defonce ^:private MirrorPad-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MirrorPad")))
(defn MirrorPad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MirrorPad"}
   (apply base/call-function @MirrorPad-fnptr* args)))

(defonce ^:private MultiBoxPrior-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MultiBoxPrior")))
(defn MultiBoxPrior
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MultiBoxPrior"}
   (apply base/call-function @MultiBoxPrior-fnptr* args)))

(defonce ^:private MultiBoxTransformLoc-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.MultiBoxTransformLoc")))
(defn MultiBoxTransformLoc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.MultiBoxTransformLoc"}
   (apply base/call-function @MultiBoxTransformLoc-fnptr* args)))

(defonce ^:private NMS-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.NMS")))
(defn NMS
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.NMS"}
   (apply base/call-function @NMS-fnptr* args)))

(defonce ^:private NdarraySize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.NdarraySize")))
(defn NdarraySize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.NdarraySize"}
   (apply base/call-function @NdarraySize-fnptr* args)))

(defonce ^:private OneHot-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.OneHot")))
(defn OneHot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.OneHot"}
   (apply base/call-function @OneHot-fnptr* args)))

(defonce ^:private PRelu-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.PRelu")))
(defn PRelu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.PRelu"}
   (apply base/call-function @PRelu-fnptr* args)))

(defonce ^:private Pad-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Pad")))
(defn Pad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Pad"}
   (apply base/call-function @Pad-fnptr* args)))

(defonce ^:private Proposal-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Proposal")))
(defn Proposal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Proposal"}
   (apply base/call-function @Proposal-fnptr* args)))

(defonce ^:private QDense-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.QDense")))
(defn QDense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.QDense"}
   (apply base/call-function @QDense-fnptr* args)))

(defonce ^:private QnnBroadcast-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.QnnBroadcast")))
(defn QnnBroadcast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.QnnBroadcast"}
   (apply base/call-function @QnnBroadcast-fnptr* args)))

(defonce ^:private QnnConcatenate-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.QnnConcatenate")))
(defn QnnConcatenate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.QnnConcatenate"}
   (apply base/call-function @QnnConcatenate-fnptr* args)))

(defonce ^:private QnnConv2D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.QnnConv2D")))
(defn QnnConv2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.QnnConv2D"}
   (apply base/call-function @QnnConv2D-fnptr* args)))

(defonce ^:private Quantize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Quantize")))
(defn Quantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Quantize"}
   (apply base/call-function @Quantize-fnptr* args)))

(defonce ^:private ROIAlign-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ROIAlign")))
(defn ROIAlign
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ROIAlign"}
   (apply base/call-function @ROIAlign-fnptr* args)))

(defonce ^:private ROIPool-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ROIPool")))
(defn ROIPool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ROIPool"}
   (apply base/call-function @ROIPool-fnptr* args)))

(defonce ^:private Reduce-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Reduce")))
(defn Reduce
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Reduce"}
   (apply base/call-function @Reduce-fnptr* args)))

(defonce ^:private Reinterpret-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Reinterpret")))
(defn Reinterpret
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Reinterpret"}
   (apply base/call-function @Reinterpret-fnptr* args)))

(defonce ^:private Repeat-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Repeat")))
(defn Repeat
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Repeat"}
   (apply base/call-function @Repeat-fnptr* args)))

(defonce ^:private Requantize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Requantize")))
(defn Requantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Requantize"}
   (apply base/call-function @Requantize-fnptr* args)))

(defonce ^:private Reshape-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Reshape")))
(defn Reshape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Reshape"}
   (apply base/call-function @Reshape-fnptr* args)))

(defonce ^:private ReshapeLike-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ReshapeLike")))
(defn ReshapeLike
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ReshapeLike"}
   (apply base/call-function @ReshapeLike-fnptr* args)))

(defonce ^:private ReshapeTensor-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ReshapeTensor")))
(defn ReshapeTensor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ReshapeTensor"}
   (apply base/call-function @ReshapeTensor-fnptr* args)))

(defonce ^:private Resize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Resize")))
(defn Resize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Resize"}
   (apply base/call-function @Resize-fnptr* args)))

(defonce ^:private Resize3d-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Resize3d")))
(defn Resize3d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Resize3d"}
   (apply base/call-function @Resize3d-fnptr* args)))

(defonce ^:private Reverse-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Reverse")))
(defn Reverse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Reverse"}
   (apply base/call-function @Reverse-fnptr* args)))

(defonce ^:private ReverseSequence-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ReverseSequence")))
(defn ReverseSequence
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ReverseSequence"}
   (apply base/call-function @ReverseSequence-fnptr* args)))

(defonce ^:private Scatter-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Scatter")))
(defn Scatter
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Scatter"}
   (apply base/call-function @Scatter-fnptr* args)))

(defonce ^:private ScatterAdd-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ScatterAdd")))
(defn ScatterAdd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ScatterAdd"}
   (apply base/call-function @ScatterAdd-fnptr* args)))

(defonce ^:private SequenceMask-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SequenceMask")))
(defn SequenceMask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SequenceMask"}
   (apply base/call-function @SequenceMask-fnptr* args)))

(defonce ^:private ShapeFuncRel-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ShapeFuncRel")))
(defn ShapeFuncRel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ShapeFuncRel"}
   (apply base/call-function @ShapeFuncRel-fnptr* args)))

(defonce ^:private ShapeOf-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.ShapeOf")))
(defn ShapeOf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.ShapeOf"}
   (apply base/call-function @ShapeOf-fnptr* args)))

(defonce ^:private SimulatedQuantize-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SimulatedQuantize")))
(defn SimulatedQuantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SimulatedQuantize"}
   (apply base/call-function @SimulatedQuantize-fnptr* args)))

(defonce ^:private SliceLike-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SliceLike")))
(defn SliceLike
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SliceLike"}
   (apply base/call-function @SliceLike-fnptr* args)))

(defonce ^:private SpaceToDepth-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SpaceToDepth")))
(defn SpaceToDepth
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SpaceToDepth"}
   (apply base/call-function @SpaceToDepth-fnptr* args)))

(defonce ^:private SparseDense-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SparseDense")))
(defn SparseDense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SparseDense"}
   (apply base/call-function @SparseDense-fnptr* args)))

(defonce ^:private SparseToDense-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SparseToDense")))
(defn SparseToDense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SparseToDense"}
   (apply base/call-function @SparseToDense-fnptr* args)))

(defonce ^:private SparseTranspose-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.SparseTranspose")))
(defn SparseTranspose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.SparseTranspose"}
   (apply base/call-function @SparseTranspose-fnptr* args)))

(defonce ^:private Split-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Split")))
(defn Split
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Split"}
   (apply base/call-function @Split-fnptr* args)))

(defonce ^:private Squeeze-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Squeeze")))
(defn Squeeze
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Squeeze"}
   (apply base/call-function @Squeeze-fnptr* args)))

(defonce ^:private Stack-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Stack")))
(defn Stack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Stack"}
   (apply base/call-function @Stack-fnptr* args)))

(defonce ^:private StridedSet-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.StridedSet")))
(defn StridedSet
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.StridedSet"}
   (apply base/call-function @StridedSet-fnptr* args)))

(defonce ^:private StridedSlice-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.StridedSlice")))
(defn StridedSlice
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.StridedSlice"}
   (apply base/call-function @StridedSlice-fnptr* args)))

(defonce ^:private Take-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Take")))
(defn Take
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Take"}
   (apply base/call-function @Take-fnptr* args)))

(defonce ^:private Tile-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Tile")))
(defn Tile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Tile"}
   (apply base/call-function @Tile-fnptr* args)))

(defonce ^:private TopK-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.TopK")))
(defn TopK
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.TopK"}
   (apply base/call-function @TopK-fnptr* args)))

(defonce ^:private Transpose-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Transpose")))
(defn Transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Transpose"}
   (apply base/call-function @Transpose-fnptr* args)))

(defonce ^:private TupleGetItem-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.TupleGetItem")))
(defn TupleGetItem
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.TupleGetItem"}
   (apply base/call-function @TupleGetItem-fnptr* args)))

(defonce ^:private UnRavelIndexRel-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.UnRavelIndexRel")))
(defn UnRavelIndexRel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.UnRavelIndexRel"}
   (apply base/call-function @UnRavelIndexRel-fnptr* args)))

(defonce ^:private UpSampling-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.UpSampling")))
(defn UpSampling
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.UpSampling"}
   (apply base/call-function @UpSampling-fnptr* args)))

(defonce ^:private UpSampling3D-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.UpSampling3D")))
(defn UpSampling3D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.UpSampling3D"}
   (apply base/call-function @UpSampling3D-fnptr* args)))

(defonce ^:private Variance-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Variance")))
(defn Variance
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Variance"}
   (apply base/call-function @Variance-fnptr* args)))

(defonce ^:private Where-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.Where")))
(defn Where
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.Where"}
   (apply base/call-function @Where-fnptr* args)))

(defonce ^:private YoloReorg-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.YoloReorg")))
(defn YoloReorg
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.YoloReorg"}
   (apply base/call-function @YoloReorg-fnptr* args)))

(defonce ^:private layout_transform-fnptr* (delay (base/name->global-function "tvm.relay.type_relation.layout_transform")))
(defn layout_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay.type_relation.layout_transform"}
   (apply base/call-function @layout_transform-fnptr* args)))

