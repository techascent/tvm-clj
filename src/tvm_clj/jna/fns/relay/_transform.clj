(ns tvm-clj.jna.fns.relay._transform
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AlterOpLayout
(let [gfn* (delay (jna-base/name->global-function "relay._transform.AlterOpLayout"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AnnotateTarget
(let [gfn* (delay (jna-base/name->global-function "relay._transform.AnnotateTarget"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BackwardFoldScaleAxis
(let [gfn* (delay (jna-base/name->global-function "relay._transform.BackwardFoldScaleAxis"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CanonicalizeCast
(let [gfn* (delay (jna-base/name->global-function "relay._transform.CanonicalizeCast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CanonicalizeOps
(let [gfn* (delay (jna-base/name->global-function "relay._transform.CanonicalizeOps"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CombineParallelBatchMatmul
(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelBatchMatmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CombineParallelConv2D
(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelConv2D"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CombineParallelDense
(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelDense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CombineParallelOpBatch
(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelOpBatch"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ConvertLayout
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ConvertLayout"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DeadCodeElimination
(let [gfn* (delay (jna-base/name->global-function "relay._transform.DeadCodeElimination"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Defunctionalization
(let [gfn* (delay (jna-base/name->global-function "relay._transform.Defunctionalization"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DenseToSparse
(let [gfn* (delay (jna-base/name->global-function "relay._transform.DenseToSparse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DynamicToStatic
(let [gfn* (delay (jna-base/name->global-function "relay._transform.DynamicToStatic"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EliminateCommonSubexpr
(let [gfn* (delay (jna-base/name->global-function "relay._transform.EliminateCommonSubexpr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EtaExpand
(let [gfn* (delay (jna-base/name->global-function "relay._transform.EtaExpand"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FastMath
(let [gfn* (delay (jna-base/name->global-function "relay._transform.FastMath"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FoldConstant
(let [gfn* (delay (jna-base/name->global-function "relay._transform.FoldConstant"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FoldScaleAxis
(let [gfn* (delay (jna-base/name->global-function "relay._transform.FoldScaleAxis"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ForwardFoldScaleAxis
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ForwardFoldScaleAxis"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FuseOps
(let [gfn* (delay (jna-base/name->global-function "relay._transform.FuseOps"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InferType
(let [gfn* (delay (jna-base/name->global-function "relay._transform.InferType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Inline
(let [gfn* (delay (jna-base/name->global-function "relay._transform.Inline"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InlinePrimitives
(let [gfn* (delay (jna-base/name->global-function "relay._transform.InlinePrimitives"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LambdaLift
(let [gfn* (delay (jna-base/name->global-function "relay._transform.LambdaLift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LazyGradientInit
(let [gfn* (delay (jna-base/name->global-function "relay._transform.LazyGradientInit"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Legalize
(let [gfn* (delay (jna-base/name->global-function "relay._transform.Legalize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MakeFunctionPass
(let [gfn* (delay (jna-base/name->global-function "relay._transform.MakeFunctionPass"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MergeCompilerRegions
(let [gfn* (delay (jna-base/name->global-function "relay._transform.MergeCompilerRegions"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MergeComposite
(let [gfn* (delay (jna-base/name->global-function "relay._transform.MergeComposite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PartialEvaluate
(let [gfn* (delay (jna-base/name->global-function "relay._transform.PartialEvaluate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PartitionGraph
(let [gfn* (delay (jna-base/name->global-function "relay._transform.PartitionGraph"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RemoveUnusedFunctions
(let [gfn* (delay (jna-base/name->global-function "relay._transform.RemoveUnusedFunctions"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RewriteDeviceAnnotation
(let [gfn* (delay (jna-base/name->global-function "relay._transform.RewriteDeviceAnnotation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SimplifyExpr
(let [gfn* (delay (jna-base/name->global-function "relay._transform.SimplifyExpr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SimplifyFCTranspose
(let [gfn* (delay (jna-base/name->global-function "relay._transform.SimplifyFCTranspose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SimplifyInference
(let [gfn* (delay (jna-base/name->global-function "relay._transform.SimplifyInference"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ToANormalForm
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToANormalForm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ToANormalFormExpr
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToANormalFormExpr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ToBasicBlockNormalForm
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToBasicBlockNormalForm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ToCPS
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToCPS"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ToGraphNormalForm
(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToGraphNormalForm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} UnCPS
(let [gfn* (delay (jna-base/name->global-function "relay._transform.UnCPS"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dedup
(let [gfn* (delay (jna-base/name->global-function "relay._transform.dedup"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} first_order_gradient
(let [gfn* (delay (jna-base/name->global-function "relay._transform.first_order_gradient"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} gradient
(let [gfn* (delay (jna-base/name->global-function "relay._transform.gradient"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} to_cps
(let [gfn* (delay (jna-base/name->global-function "relay._transform.to_cps"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} un_cps
(let [gfn* (delay (jna-base/name->global-function "relay._transform.un_cps"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

