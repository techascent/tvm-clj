(ns tvm-clj.jna.fns.tvm.relay.type_relation
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveAvgPool2D"))]
  (defn AdaptiveAvgPool2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AdaptiveAvgPool2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveAvgPool3D"))]
  (defn AdaptiveAvgPool3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AdaptiveAvgPool3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveMaxPool2D"))]
  (defn AdaptiveMaxPool2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AdaptiveMaxPool2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveMaxPool3D"))]
  (defn AdaptiveMaxPool3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AdaptiveMaxPool3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdvIndex"))]
  (defn AdvIndex
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AdvIndex"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AffineGrid"))]
  (defn AffineGrid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AffineGrid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AllocStorage"))]
  (defn AllocStorage
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AllocStorage"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AllocTensor"))]
  (defn AllocTensor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AllocTensor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Arange"))]
  (defn Arange
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Arange"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ArgReduce"))]
  (defn ArgReduce
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ArgReduce"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ArgWhere"))]
  (defn ArgWhere
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ArgWhere"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Argsort"))]
  (defn Argsort
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Argsort"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AvgPool1D"))]
  (defn AvgPool1D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AvgPool1D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AvgPool2D"))]
  (defn AvgPool2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AvgPool2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AvgPool3D"))]
  (defn AvgPool3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.AvgPool3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BatchFlatten"))]
  (defn BatchFlatten
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BatchFlatten"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BatchMatmul"))]
  (defn BatchMatmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BatchMatmul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BatchNorm"))]
  (defn BatchNorm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BatchNorm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BiasAdd"))]
  (defn BiasAdd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BiasAdd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BinaryConv2D"))]
  (defn BinaryConv2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BinaryConv2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BinaryDense"))]
  (defn BinaryDense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BinaryDense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BitPack"))]
  (defn BitPack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BitPack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BroadCastTo"))]
  (defn BroadCastTo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BroadCastTo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BroadCastToLike"))]
  (defn BroadCastToLike
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BroadCastToLike"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Broadcast"))]
  (defn Broadcast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Broadcast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BroadcastComp"))]
  (defn BroadcastComp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.BroadcastComp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Cast"))]
  (defn Cast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Cast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CastLike"))]
  (defn CastLike
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.CastLike"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CollapseSumLike"))]
  (defn CollapseSumLike
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.CollapseSumLike"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CollapseSumTo"))]
  (defn CollapseSumTo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.CollapseSumTo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Concatenate"))]
  (defn Concatenate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Concatenate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv1D"))]
  (defn Conv1D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv1D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv1DTranspose"))]
  (defn Conv1DTranspose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv1DTranspose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2D"))]
  (defn Conv2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DGemm"))]
  (defn Conv2DGemm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DGemm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DGemmWeightTransform"))]
  (defn Conv2DGemmWeightTransform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DGemmWeightTransform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DNCHWc"))]
  (defn Conv2DNCHWc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DNCHWc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DTranspose"))]
  (defn Conv2DTranspose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DTranspose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DWinograd"))]
  (defn Conv2DWinograd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DWinograd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DWinogradNNPACKWeightTransform"))]
  (defn Conv2DWinogradNNPACKWeightTransform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DWinogradNNPACKWeightTransform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DWinogradWeightTransform"))]
  (defn Conv2DWinogradWeightTransform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv2DWinogradWeightTransform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3D"))]
  (defn Conv3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3DTranspose"))]
  (defn Conv3DTranspose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv3DTranspose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3DWinograd"))]
  (defn Conv3DWinograd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv3DWinograd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3DWinogradWeightTransform"))]
  (defn Conv3DWinogradWeightTransform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Conv3DWinogradWeightTransform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Correlation"))]
  (defn Correlation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Correlation"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CropAndResize"))]
  (defn CropAndResize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.CropAndResize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CrossEntropy"))]
  (defn CrossEntropy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.CrossEntropy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Debug"))]
  (defn Debug
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Debug"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DeformableConv2D"))]
  (defn DeformableConv2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DeformableConv2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dense"))]
  (defn Dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DepthToSpace"))]
  (defn DepthToSpace
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DepthToSpace"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dequantize"))]
  (defn Dequantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Dequantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dilate"))]
  (defn Dilate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Dilate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dilation2D"))]
  (defn Dilation2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Dilation2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dropout"))]
  (defn Dropout
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Dropout"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynOneHot"))]
  (defn DynOneHot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynOneHot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynResize"))]
  (defn DynResize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynResize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynStridedSlice"))]
  (defn DynStridedSlice
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynStridedSlice"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynTopK"))]
  (defn DynTopK
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynTopK"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicBroadCastTo"))]
  (defn DynamicBroadCastTo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicBroadCastTo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicFull"))]
  (defn DynamicFull
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicFull"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicInitOp"))]
  (defn DynamicInitOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicInitOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicPad"))]
  (defn DynamicPad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicPad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicReshape"))]
  (defn DynamicReshape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicReshape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicTile"))]
  (defn DynamicTile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicTile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicUpSampling"))]
  (defn DynamicUpSampling
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicUpSampling"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicUpSampling3D"))]
  (defn DynamicUpSampling3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.DynamicUpSampling3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ExpandDims"))]
  (defn ExpandDims
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ExpandDims"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.FIFOBuffer"))]
  (defn FIFOBuffer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.FIFOBuffer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Full"))]
  (defn Full
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Full"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.FullLike"))]
  (defn FullLike
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.FullLike"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Gather"))]
  (defn Gather
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Gather"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GatherND"))]
  (defn GatherND
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.GatherND"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GetValidCount"))]
  (defn GetValidCount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.GetValidCount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GlobalAvgPool2D"))]
  (defn GlobalAvgPool2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.GlobalAvgPool2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GlobalMaxPool2D"))]
  (defn GlobalMaxPool2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.GlobalMaxPool2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GridSample"))]
  (defn GridSample
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.GridSample"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GroupNorm"))]
  (defn GroupNorm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.GroupNorm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Identity"))]
  (defn Identity
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Identity"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.IdentityCompRel"))]
  (defn IdentityCompRel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.IdentityCompRel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.InitOp"))]
  (defn InitOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.InitOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.InstanceNorm"))]
  (defn InstanceNorm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.InstanceNorm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.InvokeTVMOp"))]
  (defn InvokeTVMOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.InvokeTVMOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Kill"))]
  (defn Kill
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Kill"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.LayerNorm"))]
  (defn LayerNorm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.LayerNorm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MatrixSetDiag"))]
  (defn MatrixSetDiag
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MatrixSetDiag"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool1D"))]
  (defn MaxPool1D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MaxPool1D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool2D"))]
  (defn MaxPool2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MaxPool2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool2DGrad"))]
  (defn MaxPool2DGrad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MaxPool2DGrad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool3D"))]
  (defn MaxPool3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MaxPool3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Meshgrid"))]
  (defn Meshgrid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Meshgrid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MetaRef"))]
  (defn MetaRef
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MetaRef"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MirrorPad"))]
  (defn MirrorPad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MirrorPad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MultiBoxPrior"))]
  (defn MultiBoxPrior
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MultiBoxPrior"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MultiBoxTransformLoc"))]
  (defn MultiBoxTransformLoc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.MultiBoxTransformLoc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.NMS"))]
  (defn NMS
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.NMS"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.NdarraySize"))]
  (defn NdarraySize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.NdarraySize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.OneHot"))]
  (defn OneHot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.OneHot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.PRelu"))]
  (defn PRelu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.PRelu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Pad"))]
  (defn Pad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Pad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Proposal"))]
  (defn Proposal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Proposal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QDense"))]
  (defn QDense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.QDense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QnnBroadcast"))]
  (defn QnnBroadcast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.QnnBroadcast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QnnConcatenate"))]
  (defn QnnConcatenate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.QnnConcatenate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QnnConv2D"))]
  (defn QnnConv2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.QnnConv2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Quantize"))]
  (defn Quantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Quantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ROIAlign"))]
  (defn ROIAlign
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ROIAlign"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ROIPool"))]
  (defn ROIPool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ROIPool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reduce"))]
  (defn Reduce
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Reduce"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reinterpret"))]
  (defn Reinterpret
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Reinterpret"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Repeat"))]
  (defn Repeat
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Repeat"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Requantize"))]
  (defn Requantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Requantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reshape"))]
  (defn Reshape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Reshape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ReshapeLike"))]
  (defn ReshapeLike
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ReshapeLike"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ReshapeTensor"))]
  (defn ReshapeTensor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ReshapeTensor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Resize"))]
  (defn Resize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Resize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Resize3d"))]
  (defn Resize3d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Resize3d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reverse"))]
  (defn Reverse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Reverse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ReverseSequence"))]
  (defn ReverseSequence
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ReverseSequence"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Scatter"))]
  (defn Scatter
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Scatter"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ScatterAdd"))]
  (defn ScatterAdd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ScatterAdd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SequenceMask"))]
  (defn SequenceMask
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SequenceMask"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ShapeFuncRel"))]
  (defn ShapeFuncRel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ShapeFuncRel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ShapeOf"))]
  (defn ShapeOf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.ShapeOf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SimulatedQuantize"))]
  (defn SimulatedQuantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SimulatedQuantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SliceLike"))]
  (defn SliceLike
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SliceLike"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SpaceToDepth"))]
  (defn SpaceToDepth
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SpaceToDepth"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SparseDense"))]
  (defn SparseDense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SparseDense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SparseToDense"))]
  (defn SparseToDense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SparseToDense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SparseTranspose"))]
  (defn SparseTranspose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.SparseTranspose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Split"))]
  (defn Split
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Split"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Squeeze"))]
  (defn Squeeze
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Squeeze"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Stack"))]
  (defn Stack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Stack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.StridedSet"))]
  (defn StridedSet
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.StridedSet"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.StridedSlice"))]
  (defn StridedSlice
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.StridedSlice"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Take"))]
  (defn Take
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Take"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Tile"))]
  (defn Tile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Tile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.TopK"))]
  (defn TopK
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.TopK"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Transpose"))]
  (defn Transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.TupleGetItem"))]
  (defn TupleGetItem
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.TupleGetItem"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.UnRavelIndexRel"))]
  (defn UnRavelIndexRel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.UnRavelIndexRel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.UpSampling"))]
  (defn UpSampling
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.UpSampling"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.UpSampling3D"))]
  (defn UpSampling3D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.UpSampling3D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Variance"))]
  (defn Variance
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Variance"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Where"))]
  (defn Where
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.Where"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.YoloReorg"))]
  (defn YoloReorg
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.YoloReorg"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.layout_transform"))]
  (defn layout_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay.type_relation.layout_transform"}
     (apply jna-base/call-function @gfn* args))))

