(ns tvm-clj.jna.fns.te
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ComputeOp
(let [gfn* (delay (jna-base/name->global-function "te.ComputeOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateSchedule
(let [gfn* (delay (jna-base/name->global-function "te.CreateSchedule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateSpecializedCondition
(let [gfn* (delay (jna-base/name->global-function "te.CreateSpecializedCondition"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EnterSpecializationScope
(let [gfn* (delay (jna-base/name->global-function "te.EnterSpecializationScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ExitSpecializationScope
(let [gfn* (delay (jna-base/name->global-function "te.ExitSpecializationScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ExternOp
(let [gfn* (delay (jna-base/name->global-function "te.ExternOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetCurrentSpecialization
(let [gfn* (delay (jna-base/name->global-function "te.GetCurrentSpecialization"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Gradient
(let [gfn* (delay (jna-base/name->global-function "te.Gradient"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} HybridOp
(let [gfn* (delay (jna-base/name->global-function "te.HybridOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpGetOutput
(let [gfn* (delay (jna-base/name->global-function "te.OpGetOutput"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpInputTensors
(let [gfn* (delay (jna-base/name->global-function "te.OpInputTensors"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpNumOutputs
(let [gfn* (delay (jna-base/name->global-function "te.OpNumOutputs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Placeholder
(let [gfn* (delay (jna-base/name->global-function "te.Placeholder"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScanOp
(let [gfn* (delay (jna-base/name->global-function "te.ScanOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScheduleCacheRead
(let [gfn* (delay (jna-base/name->global-function "te.ScheduleCacheRead"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScheduleCacheWrite
(let [gfn* (delay (jna-base/name->global-function "te.ScheduleCacheWrite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScheduleCreateGroup
(let [gfn* (delay (jna-base/name->global-function "te.ScheduleCreateGroup"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScheduleNormalize
(let [gfn* (delay (jna-base/name->global-function "te.ScheduleNormalize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScheduleRFactor
(let [gfn* (delay (jna-base/name->global-function "te.ScheduleRFactor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageBind
(let [gfn* (delay (jna-base/name->global-function "te.StageBind"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageComputeAt
(let [gfn* (delay (jna-base/name->global-function "te.StageComputeAt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageComputeInline
(let [gfn* (delay (jna-base/name->global-function "te.StageComputeInline"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageComputeRoot
(let [gfn* (delay (jna-base/name->global-function "te.StageComputeRoot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageDoubleBuffer
(let [gfn* (delay (jna-base/name->global-function "te.StageDoubleBuffer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageEnvThreads
(let [gfn* (delay (jna-base/name->global-function "te.StageEnvThreads"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageFuse
(let [gfn* (delay (jna-base/name->global-function "te.StageFuse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageParallel
(let [gfn* (delay (jna-base/name->global-function "te.StageParallel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StagePragma
(let [gfn* (delay (jna-base/name->global-function "te.StagePragma"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StagePrefetch
(let [gfn* (delay (jna-base/name->global-function "te.StagePrefetch"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageReorder
(let [gfn* (delay (jna-base/name->global-function "te.StageReorder"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageSetScope
(let [gfn* (delay (jna-base/name->global-function "te.StageSetScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageSetStorePredicate
(let [gfn* (delay (jna-base/name->global-function "te.StageSetStorePredicate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageSplitByFactor
(let [gfn* (delay (jna-base/name->global-function "te.StageSplitByFactor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageSplitByNParts
(let [gfn* (delay (jna-base/name->global-function "te.StageSplitByNParts"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageStorageAlign
(let [gfn* (delay (jna-base/name->global-function "te.StageStorageAlign"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageTensorize
(let [gfn* (delay (jna-base/name->global-function "te.StageTensorize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageTile
(let [gfn* (delay (jna-base/name->global-function "te.StageTile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageUnroll
(let [gfn* (delay (jna-base/name->global-function "te.StageUnroll"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StageVectorize
(let [gfn* (delay (jna-base/name->global-function "te.StageVectorize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Tensor
(let [gfn* (delay (jna-base/name->global-function "te.Tensor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TensorComputeOp
(let [gfn* (delay (jna-base/name->global-function "te.TensorComputeOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TensorEqual
(let [gfn* (delay (jna-base/name->global-function "te.TensorEqual"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TensorHash
(let [gfn* (delay (jna-base/name->global-function "te.TensorHash"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TensorIntrin
(let [gfn* (delay (jna-base/name->global-function "te.TensorIntrin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TensorIntrinCall
(let [gfn* (delay (jna-base/name->global-function "te.TensorIntrinCall"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

