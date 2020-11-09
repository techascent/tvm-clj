(ns tvm-clj.impl.fns.relay._transform
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AlterOpLayout-fnptr* (delay (base/name->global-function "relay._transform.AlterOpLayout")))
(defn AlterOpLayout
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.AlterOpLayout"}
   (apply base/call-function @AlterOpLayout-fnptr* args)))

(defonce ^:private AnnotateTarget-fnptr* (delay (base/name->global-function "relay._transform.AnnotateTarget")))
(defn AnnotateTarget
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.AnnotateTarget"}
   (apply base/call-function @AnnotateTarget-fnptr* args)))

(defonce ^:private BackwardFoldScaleAxis-fnptr* (delay (base/name->global-function "relay._transform.BackwardFoldScaleAxis")))
(defn BackwardFoldScaleAxis
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.BackwardFoldScaleAxis"}
   (apply base/call-function @BackwardFoldScaleAxis-fnptr* args)))

(defonce ^:private CanonicalizeCast-fnptr* (delay (base/name->global-function "relay._transform.CanonicalizeCast")))
(defn CanonicalizeCast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.CanonicalizeCast"}
   (apply base/call-function @CanonicalizeCast-fnptr* args)))

(defonce ^:private CanonicalizeOps-fnptr* (delay (base/name->global-function "relay._transform.CanonicalizeOps")))
(defn CanonicalizeOps
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.CanonicalizeOps"}
   (apply base/call-function @CanonicalizeOps-fnptr* args)))

(defonce ^:private CombineParallelBatchMatmul-fnptr* (delay (base/name->global-function "relay._transform.CombineParallelBatchMatmul")))
(defn CombineParallelBatchMatmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.CombineParallelBatchMatmul"}
   (apply base/call-function @CombineParallelBatchMatmul-fnptr* args)))

(defonce ^:private CombineParallelConv2D-fnptr* (delay (base/name->global-function "relay._transform.CombineParallelConv2D")))
(defn CombineParallelConv2D
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.CombineParallelConv2D"}
   (apply base/call-function @CombineParallelConv2D-fnptr* args)))

(defonce ^:private CombineParallelDense-fnptr* (delay (base/name->global-function "relay._transform.CombineParallelDense")))
(defn CombineParallelDense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.CombineParallelDense"}
   (apply base/call-function @CombineParallelDense-fnptr* args)))

(defonce ^:private CombineParallelOpBatch-fnptr* (delay (base/name->global-function "relay._transform.CombineParallelOpBatch")))
(defn CombineParallelOpBatch
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.CombineParallelOpBatch"}
   (apply base/call-function @CombineParallelOpBatch-fnptr* args)))

(defonce ^:private ConvertLayout-fnptr* (delay (base/name->global-function "relay._transform.ConvertLayout")))
(defn ConvertLayout
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ConvertLayout"}
   (apply base/call-function @ConvertLayout-fnptr* args)))

(defonce ^:private DeadCodeElimination-fnptr* (delay (base/name->global-function "relay._transform.DeadCodeElimination")))
(defn DeadCodeElimination
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.DeadCodeElimination"}
   (apply base/call-function @DeadCodeElimination-fnptr* args)))

(defonce ^:private Defunctionalization-fnptr* (delay (base/name->global-function "relay._transform.Defunctionalization")))
(defn Defunctionalization
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.Defunctionalization"}
   (apply base/call-function @Defunctionalization-fnptr* args)))

(defonce ^:private DenseToSparse-fnptr* (delay (base/name->global-function "relay._transform.DenseToSparse")))
(defn DenseToSparse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.DenseToSparse"}
   (apply base/call-function @DenseToSparse-fnptr* args)))

(defonce ^:private DynamicToStatic-fnptr* (delay (base/name->global-function "relay._transform.DynamicToStatic")))
(defn DynamicToStatic
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.DynamicToStatic"}
   (apply base/call-function @DynamicToStatic-fnptr* args)))

(defonce ^:private EliminateCommonSubexpr-fnptr* (delay (base/name->global-function "relay._transform.EliminateCommonSubexpr")))
(defn EliminateCommonSubexpr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.EliminateCommonSubexpr"}
   (apply base/call-function @EliminateCommonSubexpr-fnptr* args)))

(defonce ^:private EtaExpand-fnptr* (delay (base/name->global-function "relay._transform.EtaExpand")))
(defn EtaExpand
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.EtaExpand"}
   (apply base/call-function @EtaExpand-fnptr* args)))

(defonce ^:private FastMath-fnptr* (delay (base/name->global-function "relay._transform.FastMath")))
(defn FastMath
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.FastMath"}
   (apply base/call-function @FastMath-fnptr* args)))

(defonce ^:private FoldConstant-fnptr* (delay (base/name->global-function "relay._transform.FoldConstant")))
(defn FoldConstant
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.FoldConstant"}
   (apply base/call-function @FoldConstant-fnptr* args)))

(defonce ^:private FoldScaleAxis-fnptr* (delay (base/name->global-function "relay._transform.FoldScaleAxis")))
(defn FoldScaleAxis
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.FoldScaleAxis"}
   (apply base/call-function @FoldScaleAxis-fnptr* args)))

(defonce ^:private ForwardFoldScaleAxis-fnptr* (delay (base/name->global-function "relay._transform.ForwardFoldScaleAxis")))
(defn ForwardFoldScaleAxis
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ForwardFoldScaleAxis"}
   (apply base/call-function @ForwardFoldScaleAxis-fnptr* args)))

(defonce ^:private FuseOps-fnptr* (delay (base/name->global-function "relay._transform.FuseOps")))
(defn FuseOps
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.FuseOps"}
   (apply base/call-function @FuseOps-fnptr* args)))

(defonce ^:private InferType-fnptr* (delay (base/name->global-function "relay._transform.InferType")))
(defn InferType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.InferType"}
   (apply base/call-function @InferType-fnptr* args)))

(defonce ^:private Inline-fnptr* (delay (base/name->global-function "relay._transform.Inline")))
(defn Inline
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.Inline"}
   (apply base/call-function @Inline-fnptr* args)))

(defonce ^:private InlinePrimitives-fnptr* (delay (base/name->global-function "relay._transform.InlinePrimitives")))
(defn InlinePrimitives
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.InlinePrimitives"}
   (apply base/call-function @InlinePrimitives-fnptr* args)))

(defonce ^:private LambdaLift-fnptr* (delay (base/name->global-function "relay._transform.LambdaLift")))
(defn LambdaLift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.LambdaLift"}
   (apply base/call-function @LambdaLift-fnptr* args)))

(defonce ^:private LazyGradientInit-fnptr* (delay (base/name->global-function "relay._transform.LazyGradientInit")))
(defn LazyGradientInit
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.LazyGradientInit"}
   (apply base/call-function @LazyGradientInit-fnptr* args)))

(defonce ^:private Legalize-fnptr* (delay (base/name->global-function "relay._transform.Legalize")))
(defn Legalize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.Legalize"}
   (apply base/call-function @Legalize-fnptr* args)))

(defonce ^:private MakeFunctionPass-fnptr* (delay (base/name->global-function "relay._transform.MakeFunctionPass")))
(defn MakeFunctionPass
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.MakeFunctionPass"}
   (apply base/call-function @MakeFunctionPass-fnptr* args)))

(defonce ^:private MergeCompilerRegions-fnptr* (delay (base/name->global-function "relay._transform.MergeCompilerRegions")))
(defn MergeCompilerRegions
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.MergeCompilerRegions"}
   (apply base/call-function @MergeCompilerRegions-fnptr* args)))

(defonce ^:private MergeComposite-fnptr* (delay (base/name->global-function "relay._transform.MergeComposite")))
(defn MergeComposite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.MergeComposite"}
   (apply base/call-function @MergeComposite-fnptr* args)))

(defonce ^:private PartialEvaluate-fnptr* (delay (base/name->global-function "relay._transform.PartialEvaluate")))
(defn PartialEvaluate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.PartialEvaluate"}
   (apply base/call-function @PartialEvaluate-fnptr* args)))

(defonce ^:private PartitionGraph-fnptr* (delay (base/name->global-function "relay._transform.PartitionGraph")))
(defn PartitionGraph
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.PartitionGraph"}
   (apply base/call-function @PartitionGraph-fnptr* args)))

(defonce ^:private RemoveUnusedFunctions-fnptr* (delay (base/name->global-function "relay._transform.RemoveUnusedFunctions")))
(defn RemoveUnusedFunctions
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.RemoveUnusedFunctions"}
   (apply base/call-function @RemoveUnusedFunctions-fnptr* args)))

(defonce ^:private RewriteDeviceAnnotation-fnptr* (delay (base/name->global-function "relay._transform.RewriteDeviceAnnotation")))
(defn RewriteDeviceAnnotation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.RewriteDeviceAnnotation"}
   (apply base/call-function @RewriteDeviceAnnotation-fnptr* args)))

(defonce ^:private SimplifyExpr-fnptr* (delay (base/name->global-function "relay._transform.SimplifyExpr")))
(defn SimplifyExpr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.SimplifyExpr"}
   (apply base/call-function @SimplifyExpr-fnptr* args)))

(defonce ^:private SimplifyFCTranspose-fnptr* (delay (base/name->global-function "relay._transform.SimplifyFCTranspose")))
(defn SimplifyFCTranspose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.SimplifyFCTranspose"}
   (apply base/call-function @SimplifyFCTranspose-fnptr* args)))

(defonce ^:private SimplifyInference-fnptr* (delay (base/name->global-function "relay._transform.SimplifyInference")))
(defn SimplifyInference
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.SimplifyInference"}
   (apply base/call-function @SimplifyInference-fnptr* args)))

(defonce ^:private ToANormalForm-fnptr* (delay (base/name->global-function "relay._transform.ToANormalForm")))
(defn ToANormalForm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ToANormalForm"}
   (apply base/call-function @ToANormalForm-fnptr* args)))

(defonce ^:private ToANormalFormExpr-fnptr* (delay (base/name->global-function "relay._transform.ToANormalFormExpr")))
(defn ToANormalFormExpr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ToANormalFormExpr"}
   (apply base/call-function @ToANormalFormExpr-fnptr* args)))

(defonce ^:private ToBasicBlockNormalForm-fnptr* (delay (base/name->global-function "relay._transform.ToBasicBlockNormalForm")))
(defn ToBasicBlockNormalForm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ToBasicBlockNormalForm"}
   (apply base/call-function @ToBasicBlockNormalForm-fnptr* args)))

(defonce ^:private ToCPS-fnptr* (delay (base/name->global-function "relay._transform.ToCPS")))
(defn ToCPS
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ToCPS"}
   (apply base/call-function @ToCPS-fnptr* args)))

(defonce ^:private ToGraphNormalForm-fnptr* (delay (base/name->global-function "relay._transform.ToGraphNormalForm")))
(defn ToGraphNormalForm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.ToGraphNormalForm"}
   (apply base/call-function @ToGraphNormalForm-fnptr* args)))

(defonce ^:private UnCPS-fnptr* (delay (base/name->global-function "relay._transform.UnCPS")))
(defn UnCPS
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.UnCPS"}
   (apply base/call-function @UnCPS-fnptr* args)))

(defonce ^:private dedup-fnptr* (delay (base/name->global-function "relay._transform.dedup")))
(defn dedup
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.dedup"}
   (apply base/call-function @dedup-fnptr* args)))

(defonce ^:private first_order_gradient-fnptr* (delay (base/name->global-function "relay._transform.first_order_gradient")))
(defn first_order_gradient
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.first_order_gradient"}
   (apply base/call-function @first_order_gradient-fnptr* args)))

(defonce ^:private gradient-fnptr* (delay (base/name->global-function "relay._transform.gradient")))
(defn gradient
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.gradient"}
   (apply base/call-function @gradient-fnptr* args)))

(defonce ^:private to_cps-fnptr* (delay (base/name->global-function "relay._transform.to_cps")))
(defn to_cps
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.to_cps"}
   (apply base/call-function @to_cps-fnptr* args)))

(defonce ^:private un_cps-fnptr* (delay (base/name->global-function "relay._transform.un_cps")))
(defn un_cps
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._transform.un_cps"}
   (apply base/call-function @un_cps-fnptr* args)))

