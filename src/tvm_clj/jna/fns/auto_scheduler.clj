(ns tvm-clj.jna.fns.auto_scheduler
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.AutoSchedule"))]
  (defn AutoSchedule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.AutoSchedule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.BuildResult"))]
  (defn BuildResult
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.BuildResult"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAG"))]
  (defn ComputeDAG
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.ComputeDAG"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAGApplyStepsFromState"))]
  (defn ComputeDAGApplyStepsFromState
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.ComputeDAGApplyStepsFromState"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAGInferBoundFromState"))]
  (defn ComputeDAGInferBoundFromState
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.ComputeDAGInferBoundFromState"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ComputeDAGPrintPythonCodeFromState"))]
  (defn ComputeDAGPrintPythonCodeFromState
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.ComputeDAGPrintPythonCodeFromState"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.CostModelPredict"))]
  (defn CostModelPredict
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.CostModelPredict"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.CostModelUpdate"))]
  (defn CostModelUpdate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.CostModelUpdate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.EmptyPolicy"))]
  (defn EmptyPolicy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.EmptyPolicy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeatureNames"))]
  (defn GetPerStoreFeatureNames
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.GetPerStoreFeatureNames"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromFile"))]
  (defn GetPerStoreFeaturesFromFile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.GetPerStoreFeaturesFromFile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromMeasurePairs"))]
  (defn GetPerStoreFeaturesFromMeasurePairs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.GetPerStoreFeaturesFromMeasurePairs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromStates"))]
  (defn GetPerStoreFeaturesFromStates
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.GetPerStoreFeaturesFromStates"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.HardwareParams"))]
  (defn HardwareParams
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.HardwareParams"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.LocalBuilder"))]
  (defn LocalBuilder
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.LocalBuilder"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.LocalRunner"))]
  (defn LocalRunner
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.LocalRunner"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.MeasureInput"))]
  (defn MeasureInput
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.MeasureInput"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.MeasureResult"))]
  (defn MeasureResult
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.MeasureResult"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.PreloadMeasuredStates"))]
  (defn PreloadMeasuredStates
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.PreloadMeasuredStates"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ProgramBuilderBuild"))]
  (defn ProgramBuilderBuild
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.ProgramBuilderBuild"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.ProgramRunnerRun"))]
  (defn ProgramRunnerRun
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.ProgramRunnerRun"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.PythonBasedModel"))]
  (defn PythonBasedModel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.PythonBasedModel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RPCRunner"))]
  (defn RPCRunner
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.RPCRunner"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RandomModel"))]
  (defn RandomModel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.RandomModel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordReader"))]
  (defn RecordReader
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.RecordReader"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordReaderReadLines"))]
  (defn RecordReaderReadLines
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.RecordReaderReadLines"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordReaderReadNext"))]
  (defn RecordReaderReadNext
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.RecordReaderReadNext"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.RecordToFile"))]
  (defn RecordToFile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.RecordToFile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SaveRecords"))]
  (defn SaveRecords
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SaveRecords"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyRunCallbacks"))]
  (defn SearchPolicyRunCallbacks
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicyRunCallbacks"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicySetTask"))]
  (defn SearchPolicySetTask
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicySetTask"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicySetVerbose"))]
  (defn SearchPolicySetVerbose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicySetVerbose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCacheReadStage"))]
  (defn SearchPolicyUtilsHasCacheReadStage
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicyUtilsHasCacheReadStage"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCacheWriteStage"))]
  (defn SearchPolicyUtilsHasCacheWriteStage
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicyUtilsHasCacheWriteStage"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction"))]
  (defn SearchPolicyUtilsHasCrossThreadReduction
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsHasRfactorStage"))]
  (defn SearchPolicyUtilsHasRfactorStage
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicyUtilsHasRfactorStage"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchPolicyUtilsIsTiled"))]
  (defn SearchPolicyUtilsIsTiled
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchPolicyUtilsIsTiled"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SearchTask"))]
  (defn SearchTask
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SearchTask"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicy"))]
  (defn SketchPolicy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SketchPolicy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicyEvolutionarySearch"))]
  (defn SketchPolicyEvolutionarySearch
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SketchPolicyEvolutionarySearch"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicyGenerateSketches"))]
  (defn SketchPolicyGenerateSketches
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SketchPolicyGenerateSketches"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.SketchPolicySampleInitialPopulation"))]
  (defn SketchPolicySampleInitialPopulation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.SketchPolicySampleInitialPopulation"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateBind"))]
  (defn StateBind
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateBind"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateCacheRead"))]
  (defn StateCacheRead
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateCacheRead"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateCacheWrite"))]
  (defn StateCacheWrite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateCacheWrite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateComputeAt"))]
  (defn StateComputeAt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateComputeAt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateComputeInline"))]
  (defn StateComputeInline
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateComputeInline"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateComputeRoot"))]
  (defn StateComputeRoot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateComputeRoot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateEqual"))]
  (defn StateEqual
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateEqual"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateFollowFusedSplit"))]
  (defn StateFollowFusedSplit
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateFollowFusedSplit"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateFollowSplit"))]
  (defn StateFollowSplit
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateFollowSplit"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateFuse"))]
  (defn StateFuse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateFuse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateParallel"))]
  (defn StateParallel
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateParallel"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StatePragma"))]
  (defn StatePragma
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StatePragma"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateReorder"))]
  (defn StateReorder
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateReorder"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateRfactor"))]
  (defn StateRfactor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateRfactor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateSplit"))]
  (defn StateSplit
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateSplit"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateStorageAlign"))]
  (defn StateStorageAlign
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateStorageAlign"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateUnroll"))]
  (defn StateUnroll
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateUnroll"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.StateVectorize"))]
  (defn StateVectorize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.StateVectorize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "auto_scheduler.TuningOptions"))]
  (defn TuningOptions
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "auto_scheduler.TuningOptions"}
     (apply jna-base/call-function @gfn* args))))

