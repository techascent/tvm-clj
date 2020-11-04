(ns tvm-clj.bindings.typenames)


(comment


  (require '[clojure.java.shell :as sh])
  (require '[clojure.string :as s])


  (def grep-results* (delay (:out (sh/sh "grep" "-hirA" "1" "_ffi.register_object" "incubator-tvm/python/"))))


  (defn results->groups
    []
    (let [lines (->> (-> @grep-results*
                         (s/split #"--"))
                     (map (fn [^String group]
                            (s/split group #"\n")))
                     (filter #(== 3 (count %))))]
      lines))


  (defn group->typename
    [[_ reg-line cls-def]]
    (if-let [[_ data] (re-find #"\"(.*)\"" reg-line)]
      data
      (second (re-find #"class ([^\(]+)" cls-def))))

  )



(def typenames
  ["Array" "Attrs" "BaseComputeOp" "ComputeOp" "DictAttrs" "EnvFunc" "ExternOp"
   "FloatImm" "FuncType" "Fuse" "GenericFunc" "GlobalTypeVar" "GlobalVar" "HybridOp"
   "IRModule" "IncompleteType" "IntImm" "Map" "Op" "PlaceholderOp" "PointerType"
   "PrimType" "Range" "ScanOp" "Schedule" "Singleton" "SourceName" "Span"
   "SpecializedCondition" "Split" "Stage" "Target" "TargetKind" "Tensor"
   "TensorComputeOp" "TensorIntrin" "TensorIntrinCall" "TupleType" "TypeCall"
   "TypeConstraint" "TypeRelation" "TypeVar" "arith.ConstIntBound"
   "arith.IntConstraints" "arith.IntConstraintsTransform" "arith.IntGroupBounds"
   "arith.IntervalSet" "arith.ModularSet" "auto_scheduler.BuildResult"
   "auto_scheduler.ComputeDAG" "auto_scheduler.CostModel" "auto_scheduler.EmptyPolicy"
   "auto_scheduler.HardwareParams" "auto_scheduler.Iterator"
   "auto_scheduler.LocalBuilder" "auto_scheduler.LocalRunner"
   "auto_scheduler.MeasureCallback" "auto_scheduler.MeasureInput"
   "auto_scheduler.MeasureResult" "auto_scheduler.PreloadMeasuredStates"
   "auto_scheduler.ProgramBuilder" "auto_scheduler.ProgramRunner"
   "auto_scheduler.PythonBasedModel" "auto_scheduler.RPCRunner"
   "auto_scheduler.RandomModel" "auto_scheduler.RecordReader"
   "auto_scheduler.RecordToFile" "auto_scheduler.SearchCallback"
   "auto_scheduler.SearchPolicy" "auto_scheduler.SearchTask"
   "auto_scheduler.SketchPolicy" "auto_scheduler.Stage" "auto_scheduler.State"
   "auto_scheduler.TuningOptions" "relay.CCacheKey" "relay.CCacheValue" "relay.Call"
   "relay.Clause" "relay.CompileEngine" "relay.Constant" "relay.Constructor"
   "relay.ConstructorValue" "relay.Function" "relay.FunctionPass" "relay.Id" "relay.If"
   "relay.Let" "relay.LoweredOutput" "relay.Match" "relay.OpImplementation"
   "relay.OpSpecialization" "relay.OpStrategy" "relay.PatternConstructor"
   "relay.PatternTuple" "relay.PatternVar" "relay.PatternWildcard" "relay.QAnnotateExpr"
   "relay.QPartitionExpr" "relay.RefCreate" "relay.RefRead" "relay.RefType"
   "relay.RefValue" "relay.RefWrite" "relay.TensorType" "relay.Tuple"
   "relay.TupleGetItem" "relay.TypeData" "relay.Var" "relay.attrs.AdaptivePool2DAttrs"
   "relay.attrs.AdaptivePool3DAttrs" "relay.attrs.AffineGridAttrs"
   "relay.attrs.AllocStorageAttrs" "relay.attrs.AllocTensorAttrs"
   "relay.attrs.ArangeAttrs" "relay.attrs.ArgsortAttrs" "relay.attrs.AvgPool1DAttrs"
   "relay.attrs.AvgPool2DAttrs" "relay.attrs.AvgPool3DAttrs"
   "relay.attrs.BatchNormAttrs" "relay.attrs.BiasAddAttrs"
   "relay.attrs.BinaryConv2DAttrs" "relay.attrs.BinaryDenseAttrs"
   "relay.attrs.BitPackAttrs" "relay.attrs.CastAttrs" "relay.attrs.CastHintAttrs"
   "relay.attrs.ClipAttrs" "relay.attrs.CompilerAttrs" "relay.attrs.ConcatenateAttrs"
   "relay.attrs.Conv1DAttrs" "relay.attrs.Conv1DTransposeAttrs"
   "relay.attrs.Conv2DAttrs" "relay.attrs.Conv2DTransposeAttrs"
   "relay.attrs.Conv2DWinogradAttrs"
   "relay.attrs.Conv2DWinogradNNPACKWeightTransformAttrs" "relay.attrs.Conv3DAttrs"
   "relay.attrs.Conv3DTransposeAttrs" "relay.attrs.Conv3DWinogradAttrs"
   "relay.attrs.ConvWinogradWeightTransformAttrs" "relay.attrs.CorrelationAttrs"
   "relay.attrs.CropAndResizeAttrs" "relay.attrs.DebugAttrs"
   "relay.attrs.DeformableConv2DAttrs" "relay.attrs.DenseAttrs"
   "relay.attrs.DeviceCopyAttrs" "relay.attrs.DilateAttrs" "relay.attrs.Dilation2DAttrs"
   "relay.attrs.DropoutAttrs" "relay.attrs.ExpandDimsAttrs"
   "relay.attrs.FIFOBufferAttrs" "relay.attrs.GatherAttrs"
   "relay.attrs.GetValidCountsAttrs" "relay.attrs.GlobalPool2DAttrs"
   "relay.attrs.GridSampleAttrs" "relay.attrs.GroupNormAttrs" "relay.attrs.InitOpAttrs"
   "relay.attrs.InstanceNormAttrs" "relay.attrs.L2NormalizeAttrs" "relay.attrs.LRNAttrs"
   "relay.attrs.LayerNormAttrs" "relay.attrs.LayoutTransformAttrs"
   "relay.attrs.LeakyReluAttrs" "relay.attrs.MaxPool1DAttrs"
   "relay.attrs.MaxPool2DAttrs" "relay.attrs.MaxPool3DAttrs" "relay.attrs.MeshgridAttrs"
   "relay.attrs.MirrorPadAttrs" "relay.attrs.MultiBoxPriorAttrs"
   "relay.attrs.MultiBoxTransformLocAttrs" "relay.attrs.NdarraySizeAttrs"
   "relay.attrs.NonMaximumSuppressionAttrs" "relay.attrs.OnDeviceAttrs"
   "relay.attrs.OneHotAttrs" "relay.attrs.PReluAttrs" "relay.attrs.PadAttrs"
   "relay.attrs.ProposalAttrs" "relay.attrs.QuantizeAttrs" "relay.attrs.ROIAlignAttrs"
   "relay.attrs.ROIPoolAttrs" "relay.attrs.ReduceAttrs" "relay.attrs.RepeatAttrs"
   "relay.attrs.RequantizeAttrs" "relay.attrs.ReshapeAttrs" "relay.attrs.Resize3dAttrs"
   "relay.attrs.ResizeAttrs" "relay.attrs.ReverseAttrs"
   "relay.attrs.ReverseSequenceAttrs" "relay.attrs.ScatterAttrs"
   "relay.attrs.SequenceMaskAttrs" "relay.attrs.ShapeFuncAttrs"
   "relay.attrs.ShapeOfAttrs" "relay.attrs.SimulatedQuantizeAttrs"
   "relay.attrs.SliceLikeAttrs" "relay.attrs.SoftmaxAttrs"
   "relay.attrs.SparseDenseAttrs" "relay.attrs.SparseToDenseAttrs"
   "relay.attrs.SparseTransposeAttrs" "relay.attrs.SplitAttrs"
   "relay.attrs.SqueezeAttrs" "relay.attrs.StackAttrs" "relay.attrs.StridedSliceAttrs"
   "relay.attrs.SubPixelAttrs" "relay.attrs.TakeAttrs" "relay.attrs.TileAttrs"
   "relay.attrs.TopkAttrs" "relay.attrs.TransposeAttrs" "relay.attrs.TupleGetItemAttrs"
   "relay.attrs.UpSampling3DAttrs" "relay.attrs.UpSamplingAttrs"
   "relay.attrs.VarianceAttrs" "relay.attrs.WithFuncIdAttrs"
   "relay.attrs.YoloReorgAttrs" "relay.dataflow_pattern."  "relay.quantize.QConfig"
   "runtime.ADT" "runtime.NDArray" "runtime.String" "tir.Add" "tir.Allocate" "tir.And"
   "tir.Any" "tir.AssertStmt" "tir.AttrStmt" "tir.BijectiveLayout" "tir.Broadcast"
   "tir.Buffer" "tir.BufferLoad" "tir.BufferRealize" "tir.BufferStore" "tir.Call"
   "tir.Cast" "tir.CommReducer" "tir.DataProducer" "tir.Div" "tir.EQ" "tir.Evaluate"
   "tir.FloorDiv" "tir.FloorMod" "tir.For" "tir.GE" "tir.GT" "tir.IfThenElse"
   "tir.IterVar" "tir.LE" "tir.LT" "tir.Let" "tir.LetStmt" "tir.Load" "tir.Max"
   "tir.Min" "tir.Mod" "tir.Mul" "tir.NE" "tir.Not" "tir.Or" "tir.Prefetch"
   "tir.PrimFunc" "tir.PrimFuncPass" "tir.ProducerLoad" "tir.ProducerRealize"
   "tir.ProducerStore" "tir.Ramp" "tir.Reduce" "tir.Select" "tir.SeqStmt" "tir.Shuffle"
   "tir.SizeVar" "tir.Store" "tir.StringImm" "tir.Sub" "tir.Var" "transform.ModulePass"
   "transform.Pass" "transform.PassContext" "transform.PassInfo" "transform.Sequential"])
