(ns tvm-clj.impl.fns.te
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ComputeOp-fnptr* (delay (base/name->global-function "te.ComputeOp")))
(defn ComputeOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ComputeOp"}
   (apply base/call-function @ComputeOp-fnptr* args)))

(defonce ^:private CreateSchedule-fnptr* (delay (base/name->global-function "te.CreateSchedule")))
(defn CreateSchedule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.CreateSchedule"}
   (apply base/call-function @CreateSchedule-fnptr* args)))

(defonce ^:private CreateSpecializedCondition-fnptr* (delay (base/name->global-function "te.CreateSpecializedCondition")))
(defn CreateSpecializedCondition
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.CreateSpecializedCondition"}
   (apply base/call-function @CreateSpecializedCondition-fnptr* args)))

(defonce ^:private EnterSpecializationScope-fnptr* (delay (base/name->global-function "te.EnterSpecializationScope")))
(defn EnterSpecializationScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.EnterSpecializationScope"}
   (apply base/call-function @EnterSpecializationScope-fnptr* args)))

(defonce ^:private ExitSpecializationScope-fnptr* (delay (base/name->global-function "te.ExitSpecializationScope")))
(defn ExitSpecializationScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ExitSpecializationScope"}
   (apply base/call-function @ExitSpecializationScope-fnptr* args)))

(defonce ^:private ExternOp-fnptr* (delay (base/name->global-function "te.ExternOp")))
(defn ExternOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ExternOp"}
   (apply base/call-function @ExternOp-fnptr* args)))

(defonce ^:private GetCurrentSpecialization-fnptr* (delay (base/name->global-function "te.GetCurrentSpecialization")))
(defn GetCurrentSpecialization
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.GetCurrentSpecialization"}
   (apply base/call-function @GetCurrentSpecialization-fnptr* args)))

(defonce ^:private Gradient-fnptr* (delay (base/name->global-function "te.Gradient")))
(defn Gradient
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.Gradient"}
   (apply base/call-function @Gradient-fnptr* args)))

(defonce ^:private HybridOp-fnptr* (delay (base/name->global-function "te.HybridOp")))
(defn HybridOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.HybridOp"}
   (apply base/call-function @HybridOp-fnptr* args)))

(defonce ^:private OpGetOutput-fnptr* (delay (base/name->global-function "te.OpGetOutput")))
(defn OpGetOutput
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.OpGetOutput"}
   (apply base/call-function @OpGetOutput-fnptr* args)))

(defonce ^:private OpInputTensors-fnptr* (delay (base/name->global-function "te.OpInputTensors")))
(defn OpInputTensors
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.OpInputTensors"}
   (apply base/call-function @OpInputTensors-fnptr* args)))

(defonce ^:private OpNumOutputs-fnptr* (delay (base/name->global-function "te.OpNumOutputs")))
(defn OpNumOutputs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.OpNumOutputs"}
   (apply base/call-function @OpNumOutputs-fnptr* args)))

(defonce ^:private Placeholder-fnptr* (delay (base/name->global-function "te.Placeholder")))
(defn Placeholder
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.Placeholder"}
   (apply base/call-function @Placeholder-fnptr* args)))

(defonce ^:private ScanOp-fnptr* (delay (base/name->global-function "te.ScanOp")))
(defn ScanOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ScanOp"}
   (apply base/call-function @ScanOp-fnptr* args)))

(defonce ^:private ScheduleCacheRead-fnptr* (delay (base/name->global-function "te.ScheduleCacheRead")))
(defn ScheduleCacheRead
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ScheduleCacheRead"}
   (apply base/call-function @ScheduleCacheRead-fnptr* args)))

(defonce ^:private ScheduleCacheWrite-fnptr* (delay (base/name->global-function "te.ScheduleCacheWrite")))
(defn ScheduleCacheWrite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ScheduleCacheWrite"}
   (apply base/call-function @ScheduleCacheWrite-fnptr* args)))

(defonce ^:private ScheduleCreateGroup-fnptr* (delay (base/name->global-function "te.ScheduleCreateGroup")))
(defn ScheduleCreateGroup
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ScheduleCreateGroup"}
   (apply base/call-function @ScheduleCreateGroup-fnptr* args)))

(defonce ^:private ScheduleNormalize-fnptr* (delay (base/name->global-function "te.ScheduleNormalize")))
(defn ScheduleNormalize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ScheduleNormalize"}
   (apply base/call-function @ScheduleNormalize-fnptr* args)))

(defonce ^:private ScheduleRFactor-fnptr* (delay (base/name->global-function "te.ScheduleRFactor")))
(defn ScheduleRFactor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.ScheduleRFactor"}
   (apply base/call-function @ScheduleRFactor-fnptr* args)))

(defonce ^:private StageBind-fnptr* (delay (base/name->global-function "te.StageBind")))
(defn StageBind
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageBind"}
   (apply base/call-function @StageBind-fnptr* args)))

(defonce ^:private StageComputeAt-fnptr* (delay (base/name->global-function "te.StageComputeAt")))
(defn StageComputeAt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageComputeAt"}
   (apply base/call-function @StageComputeAt-fnptr* args)))

(defonce ^:private StageComputeInline-fnptr* (delay (base/name->global-function "te.StageComputeInline")))
(defn StageComputeInline
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageComputeInline"}
   (apply base/call-function @StageComputeInline-fnptr* args)))

(defonce ^:private StageComputeRoot-fnptr* (delay (base/name->global-function "te.StageComputeRoot")))
(defn StageComputeRoot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageComputeRoot"}
   (apply base/call-function @StageComputeRoot-fnptr* args)))

(defonce ^:private StageDoubleBuffer-fnptr* (delay (base/name->global-function "te.StageDoubleBuffer")))
(defn StageDoubleBuffer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageDoubleBuffer"}
   (apply base/call-function @StageDoubleBuffer-fnptr* args)))

(defonce ^:private StageEnvThreads-fnptr* (delay (base/name->global-function "te.StageEnvThreads")))
(defn StageEnvThreads
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageEnvThreads"}
   (apply base/call-function @StageEnvThreads-fnptr* args)))

(defonce ^:private StageFuse-fnptr* (delay (base/name->global-function "te.StageFuse")))
(defn StageFuse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageFuse"}
   (apply base/call-function @StageFuse-fnptr* args)))

(defonce ^:private StageParallel-fnptr* (delay (base/name->global-function "te.StageParallel")))
(defn StageParallel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageParallel"}
   (apply base/call-function @StageParallel-fnptr* args)))

(defonce ^:private StagePragma-fnptr* (delay (base/name->global-function "te.StagePragma")))
(defn StagePragma
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StagePragma"}
   (apply base/call-function @StagePragma-fnptr* args)))

(defonce ^:private StagePrefetch-fnptr* (delay (base/name->global-function "te.StagePrefetch")))
(defn StagePrefetch
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StagePrefetch"}
   (apply base/call-function @StagePrefetch-fnptr* args)))

(defonce ^:private StageReorder-fnptr* (delay (base/name->global-function "te.StageReorder")))
(defn StageReorder
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageReorder"}
   (apply base/call-function @StageReorder-fnptr* args)))

(defonce ^:private StageSetScope-fnptr* (delay (base/name->global-function "te.StageSetScope")))
(defn StageSetScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageSetScope"}
   (apply base/call-function @StageSetScope-fnptr* args)))

(defonce ^:private StageSetStorePredicate-fnptr* (delay (base/name->global-function "te.StageSetStorePredicate")))
(defn StageSetStorePredicate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageSetStorePredicate"}
   (apply base/call-function @StageSetStorePredicate-fnptr* args)))

(defonce ^:private StageSplitByFactor-fnptr* (delay (base/name->global-function "te.StageSplitByFactor")))
(defn StageSplitByFactor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageSplitByFactor"}
   (apply base/call-function @StageSplitByFactor-fnptr* args)))

(defonce ^:private StageSplitByNParts-fnptr* (delay (base/name->global-function "te.StageSplitByNParts")))
(defn StageSplitByNParts
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageSplitByNParts"}
   (apply base/call-function @StageSplitByNParts-fnptr* args)))

(defonce ^:private StageStorageAlign-fnptr* (delay (base/name->global-function "te.StageStorageAlign")))
(defn StageStorageAlign
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageStorageAlign"}
   (apply base/call-function @StageStorageAlign-fnptr* args)))

(defonce ^:private StageTensorize-fnptr* (delay (base/name->global-function "te.StageTensorize")))
(defn StageTensorize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageTensorize"}
   (apply base/call-function @StageTensorize-fnptr* args)))

(defonce ^:private StageTile-fnptr* (delay (base/name->global-function "te.StageTile")))
(defn StageTile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageTile"}
   (apply base/call-function @StageTile-fnptr* args)))

(defonce ^:private StageUnroll-fnptr* (delay (base/name->global-function "te.StageUnroll")))
(defn StageUnroll
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageUnroll"}
   (apply base/call-function @StageUnroll-fnptr* args)))

(defonce ^:private StageVectorize-fnptr* (delay (base/name->global-function "te.StageVectorize")))
(defn StageVectorize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.StageVectorize"}
   (apply base/call-function @StageVectorize-fnptr* args)))

(defonce ^:private Tensor-fnptr* (delay (base/name->global-function "te.Tensor")))
(defn Tensor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.Tensor"}
   (apply base/call-function @Tensor-fnptr* args)))

(defonce ^:private TensorComputeOp-fnptr* (delay (base/name->global-function "te.TensorComputeOp")))
(defn TensorComputeOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.TensorComputeOp"}
   (apply base/call-function @TensorComputeOp-fnptr* args)))

(defonce ^:private TensorEqual-fnptr* (delay (base/name->global-function "te.TensorEqual")))
(defn TensorEqual
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.TensorEqual"}
   (apply base/call-function @TensorEqual-fnptr* args)))

(defonce ^:private TensorHash-fnptr* (delay (base/name->global-function "te.TensorHash")))
(defn TensorHash
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.TensorHash"}
   (apply base/call-function @TensorHash-fnptr* args)))

(defonce ^:private TensorIntrin-fnptr* (delay (base/name->global-function "te.TensorIntrin")))
(defn TensorIntrin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.TensorIntrin"}
   (apply base/call-function @TensorIntrin-fnptr* args)))

(defonce ^:private TensorIntrinCall-fnptr* (delay (base/name->global-function "te.TensorIntrinCall")))
(defn TensorIntrinCall
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "te.TensorIntrinCall"}
   (apply base/call-function @TensorIntrinCall-fnptr* args)))

