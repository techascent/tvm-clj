(ns tvm-clj.impl.fns.auto_scheduler
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AutoSchedule-fnptr* (delay (base/name->global-function "auto_scheduler.AutoSchedule")))
(defn AutoSchedule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.AutoSchedule"}
   (apply base/call-function @AutoSchedule-fnptr* args)))

(defonce ^:private BuildResult-fnptr* (delay (base/name->global-function "auto_scheduler.BuildResult")))
(defn BuildResult
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.BuildResult"}
   (apply base/call-function @BuildResult-fnptr* args)))

(defonce ^:private ComputeDAG-fnptr* (delay (base/name->global-function "auto_scheduler.ComputeDAG")))
(defn ComputeDAG
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.ComputeDAG"}
   (apply base/call-function @ComputeDAG-fnptr* args)))

(defonce ^:private ComputeDAGApplyStepsFromState-fnptr* (delay (base/name->global-function "auto_scheduler.ComputeDAGApplyStepsFromState")))
(defn ComputeDAGApplyStepsFromState
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.ComputeDAGApplyStepsFromState"}
   (apply base/call-function @ComputeDAGApplyStepsFromState-fnptr* args)))

(defonce ^:private ComputeDAGInferBoundFromState-fnptr* (delay (base/name->global-function "auto_scheduler.ComputeDAGInferBoundFromState")))
(defn ComputeDAGInferBoundFromState
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.ComputeDAGInferBoundFromState"}
   (apply base/call-function @ComputeDAGInferBoundFromState-fnptr* args)))

(defonce ^:private ComputeDAGPrintPythonCodeFromState-fnptr* (delay (base/name->global-function "auto_scheduler.ComputeDAGPrintPythonCodeFromState")))
(defn ComputeDAGPrintPythonCodeFromState
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.ComputeDAGPrintPythonCodeFromState"}
   (apply base/call-function @ComputeDAGPrintPythonCodeFromState-fnptr* args)))

(defonce ^:private CostModelPredict-fnptr* (delay (base/name->global-function "auto_scheduler.CostModelPredict")))
(defn CostModelPredict
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.CostModelPredict"}
   (apply base/call-function @CostModelPredict-fnptr* args)))

(defonce ^:private CostModelUpdate-fnptr* (delay (base/name->global-function "auto_scheduler.CostModelUpdate")))
(defn CostModelUpdate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.CostModelUpdate"}
   (apply base/call-function @CostModelUpdate-fnptr* args)))

(defonce ^:private EmptyPolicy-fnptr* (delay (base/name->global-function "auto_scheduler.EmptyPolicy")))
(defn EmptyPolicy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.EmptyPolicy"}
   (apply base/call-function @EmptyPolicy-fnptr* args)))

(defonce ^:private GetPerStoreFeatureNames-fnptr* (delay (base/name->global-function "auto_scheduler.GetPerStoreFeatureNames")))
(defn GetPerStoreFeatureNames
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.GetPerStoreFeatureNames"}
   (apply base/call-function @GetPerStoreFeatureNames-fnptr* args)))

(defonce ^:private GetPerStoreFeaturesFromFile-fnptr* (delay (base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromFile")))
(defn GetPerStoreFeaturesFromFile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.GetPerStoreFeaturesFromFile"}
   (apply base/call-function @GetPerStoreFeaturesFromFile-fnptr* args)))

(defonce ^:private GetPerStoreFeaturesFromMeasurePairs-fnptr* (delay (base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromMeasurePairs")))
(defn GetPerStoreFeaturesFromMeasurePairs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.GetPerStoreFeaturesFromMeasurePairs"}
   (apply base/call-function @GetPerStoreFeaturesFromMeasurePairs-fnptr* args)))

(defonce ^:private GetPerStoreFeaturesFromStates-fnptr* (delay (base/name->global-function "auto_scheduler.GetPerStoreFeaturesFromStates")))
(defn GetPerStoreFeaturesFromStates
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.GetPerStoreFeaturesFromStates"}
   (apply base/call-function @GetPerStoreFeaturesFromStates-fnptr* args)))

(defonce ^:private HardwareParams-fnptr* (delay (base/name->global-function "auto_scheduler.HardwareParams")))
(defn HardwareParams
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.HardwareParams"}
   (apply base/call-function @HardwareParams-fnptr* args)))

(defonce ^:private LocalBuilder-fnptr* (delay (base/name->global-function "auto_scheduler.LocalBuilder")))
(defn LocalBuilder
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.LocalBuilder"}
   (apply base/call-function @LocalBuilder-fnptr* args)))

(defonce ^:private LocalRunner-fnptr* (delay (base/name->global-function "auto_scheduler.LocalRunner")))
(defn LocalRunner
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.LocalRunner"}
   (apply base/call-function @LocalRunner-fnptr* args)))

(defonce ^:private MeasureInput-fnptr* (delay (base/name->global-function "auto_scheduler.MeasureInput")))
(defn MeasureInput
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.MeasureInput"}
   (apply base/call-function @MeasureInput-fnptr* args)))

(defonce ^:private MeasureResult-fnptr* (delay (base/name->global-function "auto_scheduler.MeasureResult")))
(defn MeasureResult
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.MeasureResult"}
   (apply base/call-function @MeasureResult-fnptr* args)))

(defonce ^:private PreloadMeasuredStates-fnptr* (delay (base/name->global-function "auto_scheduler.PreloadMeasuredStates")))
(defn PreloadMeasuredStates
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.PreloadMeasuredStates"}
   (apply base/call-function @PreloadMeasuredStates-fnptr* args)))

(defonce ^:private ProgramBuilderBuild-fnptr* (delay (base/name->global-function "auto_scheduler.ProgramBuilderBuild")))
(defn ProgramBuilderBuild
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.ProgramBuilderBuild"}
   (apply base/call-function @ProgramBuilderBuild-fnptr* args)))

(defonce ^:private ProgramRunnerRun-fnptr* (delay (base/name->global-function "auto_scheduler.ProgramRunnerRun")))
(defn ProgramRunnerRun
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.ProgramRunnerRun"}
   (apply base/call-function @ProgramRunnerRun-fnptr* args)))

(defonce ^:private PythonBasedModel-fnptr* (delay (base/name->global-function "auto_scheduler.PythonBasedModel")))
(defn PythonBasedModel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.PythonBasedModel"}
   (apply base/call-function @PythonBasedModel-fnptr* args)))

(defonce ^:private RPCRunner-fnptr* (delay (base/name->global-function "auto_scheduler.RPCRunner")))
(defn RPCRunner
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.RPCRunner"}
   (apply base/call-function @RPCRunner-fnptr* args)))

(defonce ^:private RandomModel-fnptr* (delay (base/name->global-function "auto_scheduler.RandomModel")))
(defn RandomModel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.RandomModel"}
   (apply base/call-function @RandomModel-fnptr* args)))

(defonce ^:private RecordReader-fnptr* (delay (base/name->global-function "auto_scheduler.RecordReader")))
(defn RecordReader
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.RecordReader"}
   (apply base/call-function @RecordReader-fnptr* args)))

(defonce ^:private RecordReaderReadLines-fnptr* (delay (base/name->global-function "auto_scheduler.RecordReaderReadLines")))
(defn RecordReaderReadLines
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.RecordReaderReadLines"}
   (apply base/call-function @RecordReaderReadLines-fnptr* args)))

(defonce ^:private RecordReaderReadNext-fnptr* (delay (base/name->global-function "auto_scheduler.RecordReaderReadNext")))
(defn RecordReaderReadNext
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.RecordReaderReadNext"}
   (apply base/call-function @RecordReaderReadNext-fnptr* args)))

(defonce ^:private RecordToFile-fnptr* (delay (base/name->global-function "auto_scheduler.RecordToFile")))
(defn RecordToFile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.RecordToFile"}
   (apply base/call-function @RecordToFile-fnptr* args)))

(defonce ^:private SaveRecords-fnptr* (delay (base/name->global-function "auto_scheduler.SaveRecords")))
(defn SaveRecords
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SaveRecords"}
   (apply base/call-function @SaveRecords-fnptr* args)))

(defonce ^:private SearchPolicyRunCallbacks-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicyRunCallbacks")))
(defn SearchPolicyRunCallbacks
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicyRunCallbacks"}
   (apply base/call-function @SearchPolicyRunCallbacks-fnptr* args)))

(defonce ^:private SearchPolicySetTask-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicySetTask")))
(defn SearchPolicySetTask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicySetTask"}
   (apply base/call-function @SearchPolicySetTask-fnptr* args)))

(defonce ^:private SearchPolicySetVerbose-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicySetVerbose")))
(defn SearchPolicySetVerbose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicySetVerbose"}
   (apply base/call-function @SearchPolicySetVerbose-fnptr* args)))

(defonce ^:private SearchPolicyUtilsHasCacheReadStage-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCacheReadStage")))
(defn SearchPolicyUtilsHasCacheReadStage
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicyUtilsHasCacheReadStage"}
   (apply base/call-function @SearchPolicyUtilsHasCacheReadStage-fnptr* args)))

(defonce ^:private SearchPolicyUtilsHasCacheWriteStage-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCacheWriteStage")))
(defn SearchPolicyUtilsHasCacheWriteStage
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicyUtilsHasCacheWriteStage"}
   (apply base/call-function @SearchPolicyUtilsHasCacheWriteStage-fnptr* args)))

(defonce ^:private SearchPolicyUtilsHasCrossThreadReduction-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction")))
(defn SearchPolicyUtilsHasCrossThreadReduction
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicyUtilsHasCrossThreadReduction"}
   (apply base/call-function @SearchPolicyUtilsHasCrossThreadReduction-fnptr* args)))

(defonce ^:private SearchPolicyUtilsHasRfactorStage-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicyUtilsHasRfactorStage")))
(defn SearchPolicyUtilsHasRfactorStage
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicyUtilsHasRfactorStage"}
   (apply base/call-function @SearchPolicyUtilsHasRfactorStage-fnptr* args)))

(defonce ^:private SearchPolicyUtilsIsTiled-fnptr* (delay (base/name->global-function "auto_scheduler.SearchPolicyUtilsIsTiled")))
(defn SearchPolicyUtilsIsTiled
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchPolicyUtilsIsTiled"}
   (apply base/call-function @SearchPolicyUtilsIsTiled-fnptr* args)))

(defonce ^:private SearchTask-fnptr* (delay (base/name->global-function "auto_scheduler.SearchTask")))
(defn SearchTask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SearchTask"}
   (apply base/call-function @SearchTask-fnptr* args)))

(defonce ^:private SketchPolicy-fnptr* (delay (base/name->global-function "auto_scheduler.SketchPolicy")))
(defn SketchPolicy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SketchPolicy"}
   (apply base/call-function @SketchPolicy-fnptr* args)))

(defonce ^:private SketchPolicyEvolutionarySearch-fnptr* (delay (base/name->global-function "auto_scheduler.SketchPolicyEvolutionarySearch")))
(defn SketchPolicyEvolutionarySearch
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SketchPolicyEvolutionarySearch"}
   (apply base/call-function @SketchPolicyEvolutionarySearch-fnptr* args)))

(defonce ^:private SketchPolicyGenerateSketches-fnptr* (delay (base/name->global-function "auto_scheduler.SketchPolicyGenerateSketches")))
(defn SketchPolicyGenerateSketches
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SketchPolicyGenerateSketches"}
   (apply base/call-function @SketchPolicyGenerateSketches-fnptr* args)))

(defonce ^:private SketchPolicySampleInitialPopulation-fnptr* (delay (base/name->global-function "auto_scheduler.SketchPolicySampleInitialPopulation")))
(defn SketchPolicySampleInitialPopulation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.SketchPolicySampleInitialPopulation"}
   (apply base/call-function @SketchPolicySampleInitialPopulation-fnptr* args)))

(defonce ^:private StateBind-fnptr* (delay (base/name->global-function "auto_scheduler.StateBind")))
(defn StateBind
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateBind"}
   (apply base/call-function @StateBind-fnptr* args)))

(defonce ^:private StateCacheRead-fnptr* (delay (base/name->global-function "auto_scheduler.StateCacheRead")))
(defn StateCacheRead
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateCacheRead"}
   (apply base/call-function @StateCacheRead-fnptr* args)))

(defonce ^:private StateCacheWrite-fnptr* (delay (base/name->global-function "auto_scheduler.StateCacheWrite")))
(defn StateCacheWrite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateCacheWrite"}
   (apply base/call-function @StateCacheWrite-fnptr* args)))

(defonce ^:private StateComputeAt-fnptr* (delay (base/name->global-function "auto_scheduler.StateComputeAt")))
(defn StateComputeAt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateComputeAt"}
   (apply base/call-function @StateComputeAt-fnptr* args)))

(defonce ^:private StateComputeInline-fnptr* (delay (base/name->global-function "auto_scheduler.StateComputeInline")))
(defn StateComputeInline
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateComputeInline"}
   (apply base/call-function @StateComputeInline-fnptr* args)))

(defonce ^:private StateComputeRoot-fnptr* (delay (base/name->global-function "auto_scheduler.StateComputeRoot")))
(defn StateComputeRoot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateComputeRoot"}
   (apply base/call-function @StateComputeRoot-fnptr* args)))

(defonce ^:private StateEqual-fnptr* (delay (base/name->global-function "auto_scheduler.StateEqual")))
(defn StateEqual
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateEqual"}
   (apply base/call-function @StateEqual-fnptr* args)))

(defonce ^:private StateFollowFusedSplit-fnptr* (delay (base/name->global-function "auto_scheduler.StateFollowFusedSplit")))
(defn StateFollowFusedSplit
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateFollowFusedSplit"}
   (apply base/call-function @StateFollowFusedSplit-fnptr* args)))

(defonce ^:private StateFollowSplit-fnptr* (delay (base/name->global-function "auto_scheduler.StateFollowSplit")))
(defn StateFollowSplit
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateFollowSplit"}
   (apply base/call-function @StateFollowSplit-fnptr* args)))

(defonce ^:private StateFuse-fnptr* (delay (base/name->global-function "auto_scheduler.StateFuse")))
(defn StateFuse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateFuse"}
   (apply base/call-function @StateFuse-fnptr* args)))

(defonce ^:private StateParallel-fnptr* (delay (base/name->global-function "auto_scheduler.StateParallel")))
(defn StateParallel
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateParallel"}
   (apply base/call-function @StateParallel-fnptr* args)))

(defonce ^:private StatePragma-fnptr* (delay (base/name->global-function "auto_scheduler.StatePragma")))
(defn StatePragma
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StatePragma"}
   (apply base/call-function @StatePragma-fnptr* args)))

(defonce ^:private StateReorder-fnptr* (delay (base/name->global-function "auto_scheduler.StateReorder")))
(defn StateReorder
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateReorder"}
   (apply base/call-function @StateReorder-fnptr* args)))

(defonce ^:private StateRfactor-fnptr* (delay (base/name->global-function "auto_scheduler.StateRfactor")))
(defn StateRfactor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateRfactor"}
   (apply base/call-function @StateRfactor-fnptr* args)))

(defonce ^:private StateSplit-fnptr* (delay (base/name->global-function "auto_scheduler.StateSplit")))
(defn StateSplit
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateSplit"}
   (apply base/call-function @StateSplit-fnptr* args)))

(defonce ^:private StateStorageAlign-fnptr* (delay (base/name->global-function "auto_scheduler.StateStorageAlign")))
(defn StateStorageAlign
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateStorageAlign"}
   (apply base/call-function @StateStorageAlign-fnptr* args)))

(defonce ^:private StateUnroll-fnptr* (delay (base/name->global-function "auto_scheduler.StateUnroll")))
(defn StateUnroll
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateUnroll"}
   (apply base/call-function @StateUnroll-fnptr* args)))

(defonce ^:private StateVectorize-fnptr* (delay (base/name->global-function "auto_scheduler.StateVectorize")))
(defn StateVectorize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.StateVectorize"}
   (apply base/call-function @StateVectorize-fnptr* args)))

(defonce ^:private TuningOptions-fnptr* (delay (base/name->global-function "auto_scheduler.TuningOptions")))
(defn TuningOptions
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "auto_scheduler.TuningOptions"}
   (apply base/call-function @TuningOptions-fnptr* args)))

