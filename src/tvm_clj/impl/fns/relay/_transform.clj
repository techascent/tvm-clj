(ns tvm-clj.jna.fns.relay._transform
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.AlterOpLayout"))]
  (defn AlterOpLayout
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.AlterOpLayout"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.AnnotateTarget"))]
  (defn AnnotateTarget
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.AnnotateTarget"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.BackwardFoldScaleAxis"))]
  (defn BackwardFoldScaleAxis
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.BackwardFoldScaleAxis"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.CanonicalizeCast"))]
  (defn CanonicalizeCast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.CanonicalizeCast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.CanonicalizeOps"))]
  (defn CanonicalizeOps
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.CanonicalizeOps"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelBatchMatmul"))]
  (defn CombineParallelBatchMatmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.CombineParallelBatchMatmul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelConv2D"))]
  (defn CombineParallelConv2D
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.CombineParallelConv2D"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelDense"))]
  (defn CombineParallelDense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.CombineParallelDense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.CombineParallelOpBatch"))]
  (defn CombineParallelOpBatch
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.CombineParallelOpBatch"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ConvertLayout"))]
  (defn ConvertLayout
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ConvertLayout"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.DeadCodeElimination"))]
  (defn DeadCodeElimination
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.DeadCodeElimination"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.Defunctionalization"))]
  (defn Defunctionalization
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.Defunctionalization"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.DenseToSparse"))]
  (defn DenseToSparse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.DenseToSparse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.DynamicToStatic"))]
  (defn DynamicToStatic
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.DynamicToStatic"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.EliminateCommonSubexpr"))]
  (defn EliminateCommonSubexpr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.EliminateCommonSubexpr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.EtaExpand"))]
  (defn EtaExpand
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.EtaExpand"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.FastMath"))]
  (defn FastMath
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.FastMath"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.FoldConstant"))]
  (defn FoldConstant
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.FoldConstant"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.FoldScaleAxis"))]
  (defn FoldScaleAxis
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.FoldScaleAxis"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ForwardFoldScaleAxis"))]
  (defn ForwardFoldScaleAxis
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ForwardFoldScaleAxis"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.FuseOps"))]
  (defn FuseOps
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.FuseOps"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.InferType"))]
  (defn InferType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.InferType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.Inline"))]
  (defn Inline
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.Inline"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.InlinePrimitives"))]
  (defn InlinePrimitives
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.InlinePrimitives"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.LambdaLift"))]
  (defn LambdaLift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.LambdaLift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.LazyGradientInit"))]
  (defn LazyGradientInit
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.LazyGradientInit"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.Legalize"))]
  (defn Legalize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.Legalize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.MakeFunctionPass"))]
  (defn MakeFunctionPass
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.MakeFunctionPass"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.MergeCompilerRegions"))]
  (defn MergeCompilerRegions
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.MergeCompilerRegions"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.MergeComposite"))]
  (defn MergeComposite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.MergeComposite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.PartialEvaluate"))]
  (defn PartialEvaluate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.PartialEvaluate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.PartitionGraph"))]
  (defn PartitionGraph
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.PartitionGraph"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.RemoveUnusedFunctions"))]
  (defn RemoveUnusedFunctions
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.RemoveUnusedFunctions"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.RewriteDeviceAnnotation"))]
  (defn RewriteDeviceAnnotation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.RewriteDeviceAnnotation"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.SimplifyExpr"))]
  (defn SimplifyExpr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.SimplifyExpr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.SimplifyFCTranspose"))]
  (defn SimplifyFCTranspose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.SimplifyFCTranspose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.SimplifyInference"))]
  (defn SimplifyInference
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.SimplifyInference"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToANormalForm"))]
  (defn ToANormalForm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ToANormalForm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToANormalFormExpr"))]
  (defn ToANormalFormExpr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ToANormalFormExpr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToBasicBlockNormalForm"))]
  (defn ToBasicBlockNormalForm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ToBasicBlockNormalForm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToCPS"))]
  (defn ToCPS
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ToCPS"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.ToGraphNormalForm"))]
  (defn ToGraphNormalForm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.ToGraphNormalForm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.UnCPS"))]
  (defn UnCPS
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.UnCPS"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.dedup"))]
  (defn dedup
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.dedup"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.first_order_gradient"))]
  (defn first_order_gradient
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.first_order_gradient"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.gradient"))]
  (defn gradient
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.gradient"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.to_cps"))]
  (defn to_cps
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.to_cps"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._transform.un_cps"))]
  (defn un_cps
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._transform.un_cps"}
     (apply jna-base/call-function @gfn* args))))

