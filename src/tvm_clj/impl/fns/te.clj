(ns tvm-clj.jna.fns.te
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "te.ComputeOp"))]
  (defn ComputeOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ComputeOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.CreateSchedule"))]
  (defn CreateSchedule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.CreateSchedule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.CreateSpecializedCondition"))]
  (defn CreateSpecializedCondition
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.CreateSpecializedCondition"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.EnterSpecializationScope"))]
  (defn EnterSpecializationScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.EnterSpecializationScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ExitSpecializationScope"))]
  (defn ExitSpecializationScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ExitSpecializationScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ExternOp"))]
  (defn ExternOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ExternOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.GetCurrentSpecialization"))]
  (defn GetCurrentSpecialization
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.GetCurrentSpecialization"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.Gradient"))]
  (defn Gradient
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.Gradient"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.HybridOp"))]
  (defn HybridOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.HybridOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.OpGetOutput"))]
  (defn OpGetOutput
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.OpGetOutput"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.OpInputTensors"))]
  (defn OpInputTensors
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.OpInputTensors"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.OpNumOutputs"))]
  (defn OpNumOutputs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.OpNumOutputs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.Placeholder"))]
  (defn Placeholder
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.Placeholder"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ScanOp"))]
  (defn ScanOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ScanOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ScheduleCacheRead"))]
  (defn ScheduleCacheRead
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ScheduleCacheRead"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ScheduleCacheWrite"))]
  (defn ScheduleCacheWrite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ScheduleCacheWrite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ScheduleCreateGroup"))]
  (defn ScheduleCreateGroup
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ScheduleCreateGroup"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ScheduleNormalize"))]
  (defn ScheduleNormalize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ScheduleNormalize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.ScheduleRFactor"))]
  (defn ScheduleRFactor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.ScheduleRFactor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageBind"))]
  (defn StageBind
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageBind"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageComputeAt"))]
  (defn StageComputeAt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageComputeAt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageComputeInline"))]
  (defn StageComputeInline
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageComputeInline"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageComputeRoot"))]
  (defn StageComputeRoot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageComputeRoot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageDoubleBuffer"))]
  (defn StageDoubleBuffer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageDoubleBuffer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageEnvThreads"))]
  (defn StageEnvThreads
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageEnvThreads"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageFuse"))]
  (defn StageFuse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageFuse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageParallel"))]
  (defn StageParallel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageParallel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StagePragma"))]
  (defn StagePragma
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StagePragma"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StagePrefetch"))]
  (defn StagePrefetch
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StagePrefetch"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageReorder"))]
  (defn StageReorder
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageReorder"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageSetScope"))]
  (defn StageSetScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageSetScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageSetStorePredicate"))]
  (defn StageSetStorePredicate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageSetStorePredicate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageSplitByFactor"))]
  (defn StageSplitByFactor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageSplitByFactor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageSplitByNParts"))]
  (defn StageSplitByNParts
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageSplitByNParts"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageStorageAlign"))]
  (defn StageStorageAlign
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageStorageAlign"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageTensorize"))]
  (defn StageTensorize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageTensorize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageTile"))]
  (defn StageTile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageTile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageUnroll"))]
  (defn StageUnroll
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageUnroll"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.StageVectorize"))]
  (defn StageVectorize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.StageVectorize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.Tensor"))]
  (defn Tensor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.Tensor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.TensorComputeOp"))]
  (defn TensorComputeOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.TensorComputeOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.TensorEqual"))]
  (defn TensorEqual
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.TensorEqual"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.TensorHash"))]
  (defn TensorHash
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.TensorHash"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.TensorIntrin"))]
  (defn TensorIntrin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.TensorIntrin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "te.TensorIntrinCall"))]
  (defn TensorIntrinCall
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "te.TensorIntrinCall"}
     (apply jna-base/call-function @gfn* args))))

