(ns tvm-clj.jna.fns.tvm.relay.type_relation
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AdaptiveAvgPool2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveAvgPool2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AdaptiveAvgPool3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveAvgPool3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AdaptiveMaxPool2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveMaxPool2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AdaptiveMaxPool3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdaptiveMaxPool3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AdvIndex
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AdvIndex"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AffineGrid
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AffineGrid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AllocStorage
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AllocStorage"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AllocTensor
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AllocTensor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Arange
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Arange"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ArgReduce
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ArgReduce"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ArgWhere
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ArgWhere"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Argsort
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Argsort"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AvgPool1D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AvgPool1D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AvgPool2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AvgPool2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AvgPool3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.AvgPool3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BatchFlatten
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BatchFlatten"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BatchMatmul
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BatchMatmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BatchNorm
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BatchNorm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BiasAdd
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BiasAdd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BinaryConv2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BinaryConv2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BinaryDense
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BinaryDense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BitPack
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BitPack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BroadCastTo
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BroadCastTo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BroadCastToLike
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BroadCastToLike"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Broadcast
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Broadcast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BroadcastComp
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.BroadcastComp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Cast
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Cast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CastLike
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CastLike"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CollapseSumLike
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CollapseSumLike"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CollapseSumTo
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CollapseSumTo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Concatenate
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Concatenate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv1D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv1D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv1DTranspose
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv1DTranspose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DGemm
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DGemm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DGemmWeightTransform
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DGemmWeightTransform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DNCHWc
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DNCHWc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DTranspose
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DTranspose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DWinograd
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DWinograd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DWinogradNNPACKWeightTransform
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DWinogradNNPACKWeightTransform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv2DWinogradWeightTransform
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv2DWinogradWeightTransform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv3DTranspose
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3DTranspose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv3DWinograd
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3DWinograd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Conv3DWinogradWeightTransform
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Conv3DWinogradWeightTransform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Correlation
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Correlation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CropAndResize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CropAndResize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CrossEntropy
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.CrossEntropy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Debug
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Debug"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DeformableConv2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DeformableConv2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Dense
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DepthToSpace
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DepthToSpace"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Dequantize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dequantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Dilate
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dilate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Dilation2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dilation2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Dropout
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Dropout"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynOneHot
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynOneHot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynResize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynResize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynStridedSlice
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynStridedSlice"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynTopK
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynTopK"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicBroadCastTo
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicBroadCastTo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicFull
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicFull"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicInitOp
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicInitOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicPad
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicPad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicReshape
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicReshape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicTile
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicTile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicUpSampling
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicUpSampling"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicUpSampling3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.DynamicUpSampling3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ExpandDims
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ExpandDims"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FIFOBuffer
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.FIFOBuffer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Full
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Full"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FullLike
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.FullLike"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Gather
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Gather"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GatherND
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GatherND"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetValidCount
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GetValidCount"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GlobalAvgPool2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GlobalAvgPool2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GlobalMaxPool2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GlobalMaxPool2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GridSample
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GridSample"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GroupNorm
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.GroupNorm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Identity
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Identity"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IdentityCompRel
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.IdentityCompRel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InitOp
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.InitOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InstanceNorm
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.InstanceNorm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InvokeTVMOp
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.InvokeTVMOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Kill
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Kill"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LayerNorm
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.LayerNorm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MatrixSetDiag
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MatrixSetDiag"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MaxPool1D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool1D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MaxPool2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MaxPool2DGrad
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool2DGrad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MaxPool3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MaxPool3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Meshgrid
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Meshgrid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MetaRef
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MetaRef"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MirrorPad
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MirrorPad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MultiBoxPrior
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MultiBoxPrior"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MultiBoxTransformLoc
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.MultiBoxTransformLoc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NMS
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.NMS"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NdarraySize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.NdarraySize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OneHot
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.OneHot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PRelu
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.PRelu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Pad
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Pad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Proposal
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Proposal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QDense
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QDense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QnnBroadcast
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QnnBroadcast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QnnConcatenate
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QnnConcatenate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QnnConv2D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.QnnConv2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Quantize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Quantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ROIAlign
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ROIAlign"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ROIPool
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ROIPool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Reduce
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reduce"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Reinterpret
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reinterpret"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Repeat
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Repeat"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Requantize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Requantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Reshape
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reshape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ReshapeLike
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ReshapeLike"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ReshapeTensor
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ReshapeTensor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Resize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Resize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Resize3d
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Resize3d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Reverse
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Reverse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ReverseSequence
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ReverseSequence"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Scatter
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Scatter"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScatterAdd
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ScatterAdd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SequenceMask
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SequenceMask"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ShapeFuncRel
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ShapeFuncRel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ShapeOf
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.ShapeOf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SimulatedQuantize
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SimulatedQuantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SliceLike
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SliceLike"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SpaceToDepth
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SpaceToDepth"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SparseDense
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SparseDense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SparseToDense
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SparseToDense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SparseTranspose
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.SparseTranspose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Split
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Split"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Squeeze
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Squeeze"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Stack
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Stack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StridedSet
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.StridedSet"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StridedSlice
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.StridedSlice"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Take
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Take"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Tile
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Tile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TopK
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.TopK"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Transpose
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TupleGetItem
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.TupleGetItem"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} UnRavelIndexRel
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.UnRavelIndexRel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} UpSampling
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.UpSampling"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} UpSampling3D
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.UpSampling3D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Variance
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Variance"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Where
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.Where"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} YoloReorg
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.YoloReorg"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} layout_transform
(let [gfn* (delay (jna-base/name->global-function "tvm.relay.type_relation.layout_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

