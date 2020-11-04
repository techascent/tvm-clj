(ns tvm-clj.jna.fns.auto_scheduler
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AutoSchedule
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.AutoSchedule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BuildResult
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.BuildResult"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ComputeDAG
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAG"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ComputeDAGApplyStepsFromState
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAGApplyStepsFromState"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ComputeDAGInferBoundFromState
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAGInferBoundFromState"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ComputeDAGPrintPythonCodeFromState
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAGPrintPythonCodeFromState"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CostModelPredict
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.CostModelPredict"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CostModelUpdate
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.CostModelUpdate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EmptyPolicy
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.EmptyPolicy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetPerStoreFeatureNames
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeatureNames"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetPerStoreFeaturesFromFile
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromFile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetPerStoreFeaturesFromMeasurePairs
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromMeasurePairs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetPerStoreFeaturesFromStates
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromStates"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} HardwareParams
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.HardwareParams"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LocalBuilder
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.LocalBuilder"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LocalRunner
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.LocalRunner"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MeasureInput
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.MeasureInput"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MeasureResult
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.MeasureResult"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PreloadMeasuredStates
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.PreloadMeasuredStates"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ProgramBuilderBuild
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ProgramBuilderBuild"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ProgramRunnerRun
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ProgramRunnerRun"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PythonBasedModel
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.PythonBasedModel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RPCRunner
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RPCRunner"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RandomModel
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RandomModel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RecordReader
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordReader"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RecordReaderReadLines
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordReaderReadLines"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RecordReaderReadNext
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordReaderReadNext"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RecordToFile
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordToFile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SaveRecords
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SaveRecords"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicyRunCallbacks
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyRunCallbacks"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicySetTask
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicySetTask"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicySetVerbose
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicySetVerbose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicyUtilsHasCacheReadStage
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCacheReadStage"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicyUtilsHasCacheWriteStage
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCacheWriteStage"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicyUtilsHasCrossThreadReduction
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicyUtilsHasRfactorStage
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasRfactorStage"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchPolicyUtilsIsTiled
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsIsTiled"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SearchTask
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchTask"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SketchPolicy
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SketchPolicyEvolutionarySearch
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicyEvolutionarySearch"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SketchPolicyGenerateSketches
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicyGenerateSketches"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SketchPolicySampleInitialPopulation
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicySampleInitialPopulation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateBind
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateBind"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateCacheRead
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateCacheRead"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateCacheWrite
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateCacheWrite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateComputeAt
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateComputeAt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateComputeInline
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateComputeInline"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateComputeRoot
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateComputeRoot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateEqual
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateEqual"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateFollowFusedSplit
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateFollowFusedSplit"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateFollowSplit
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateFollowSplit"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateFuse
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateFuse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateParallel
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateParallel"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StatePragma
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StatePragma"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateReorder
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateReorder"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateRfactor
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateRfactor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateSplit
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateSplit"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateStorageAlign
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateStorageAlign"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateUnroll
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateUnroll"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StateVectorize
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateVectorize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TuningOptions
(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.TuningOptions"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

