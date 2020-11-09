(ns tvm-clj.impl.fns.tir.transform
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private BF16CastElimination-fnptr* (delay (base/name->global-function "tir.transform.BF16CastElimination")))
(defn BF16CastElimination
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.BF16CastElimination"}
   (apply base/call-function @BF16CastElimination-fnptr* args)))

(defonce ^:private BF16Legalize-fnptr* (delay (base/name->global-function "tir.transform.BF16Legalize")))
(defn BF16Legalize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.BF16Legalize"}
   (apply base/call-function @BF16Legalize-fnptr* args)))

(defonce ^:private BF16Promote-fnptr* (delay (base/name->global-function "tir.transform.BF16Promote")))
(defn BF16Promote
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.BF16Promote"}
   (apply base/call-function @BF16Promote-fnptr* args)))

(defonce ^:private BF16TypeLowering-fnptr* (delay (base/name->global-function "tir.transform.BF16TypeLowering")))
(defn BF16TypeLowering
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.BF16TypeLowering"}
   (apply base/call-function @BF16TypeLowering-fnptr* args)))

(defonce ^:private CoProcSync-fnptr* (delay (base/name->global-function "tir.transform.CoProcSync")))
(defn CoProcSync
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.CoProcSync"}
   (apply base/call-function @CoProcSync-fnptr* args)))

(defonce ^:private CombineContextCall-fnptr* (delay (base/name->global-function "tir.transform.CombineContextCall")))
(defn CombineContextCall
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.CombineContextCall"}
   (apply base/call-function @CombineContextCall-fnptr* args)))

(defonce ^:private CreatePrimFuncPass-fnptr* (delay (base/name->global-function "tir.transform.CreatePrimFuncPass")))
(defn CreatePrimFuncPass
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.CreatePrimFuncPass"}
   (apply base/call-function @CreatePrimFuncPass-fnptr* args)))

(defonce ^:private DecorateDeviceScope-fnptr* (delay (base/name->global-function "tir.transform.DecorateDeviceScope")))
(defn DecorateDeviceScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.DecorateDeviceScope"}
   (apply base/call-function @DecorateDeviceScope-fnptr* args)))

(defonce ^:private HoistIfThenElse-fnptr* (delay (base/name->global-function "tir.transform.HoistIfThenElse")))
(defn HoistIfThenElse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.HoistIfThenElse"}
   (apply base/call-function @HoistIfThenElse-fnptr* args)))

(defonce ^:private HoistIfThenElseBasic-fnptr* (delay (base/name->global-function "tir.transform.HoistIfThenElseBasic")))
(defn HoistIfThenElseBasic
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.HoistIfThenElseBasic"}
   (apply base/call-function @HoistIfThenElseBasic-fnptr* args)))

(defonce ^:private InferFragment-fnptr* (delay (base/name->global-function "tir.transform.InferFragment")))
(defn InferFragment
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.InferFragment"}
   (apply base/call-function @InferFragment-fnptr* args)))

(defonce ^:private InjectCopyIntrin-fnptr* (delay (base/name->global-function "tir.transform.InjectCopyIntrin")))
(defn InjectCopyIntrin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.InjectCopyIntrin"}
   (apply base/call-function @InjectCopyIntrin-fnptr* args)))

(defonce ^:private InjectDoubleBuffer-fnptr* (delay (base/name->global-function "tir.transform.InjectDoubleBuffer")))
(defn InjectDoubleBuffer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.InjectDoubleBuffer"}
   (apply base/call-function @InjectDoubleBuffer-fnptr* args)))

(defonce ^:private InjectPrefetch-fnptr* (delay (base/name->global-function "tir.transform.InjectPrefetch")))
(defn InjectPrefetch
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.InjectPrefetch"}
   (apply base/call-function @InjectPrefetch-fnptr* args)))

(defonce ^:private InjectVirtualThread-fnptr* (delay (base/name->global-function "tir.transform.InjectVirtualThread")))
(defn InjectVirtualThread
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.InjectVirtualThread"}
   (apply base/call-function @InjectVirtualThread-fnptr* args)))

(defonce ^:private InstrumentBoundCheckers-fnptr* (delay (base/name->global-function "tir.transform.InstrumentBoundCheckers")))
(defn InstrumentBoundCheckers
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.InstrumentBoundCheckers"}
   (apply base/call-function @InstrumentBoundCheckers-fnptr* args)))

(defonce ^:private LiftAttrScope-fnptr* (delay (base/name->global-function "tir.transform.LiftAttrScope")))
(defn LiftAttrScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LiftAttrScope"}
   (apply base/call-function @LiftAttrScope-fnptr* args)))

(defonce ^:private LoopPartition-fnptr* (delay (base/name->global-function "tir.transform.LoopPartition")))
(defn LoopPartition
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LoopPartition"}
   (apply base/call-function @LoopPartition-fnptr* args)))

(defonce ^:private LowerCustomDatatypes-fnptr* (delay (base/name->global-function "tir.transform.LowerCustomDatatypes")))
(defn LowerCustomDatatypes
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LowerCustomDatatypes"}
   (apply base/call-function @LowerCustomDatatypes-fnptr* args)))

(defonce ^:private LowerDeviceStorageAccessInfo-fnptr* (delay (base/name->global-function "tir.transform.LowerDeviceStorageAccessInfo")))
(defn LowerDeviceStorageAccessInfo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LowerDeviceStorageAccessInfo"}
   (apply base/call-function @LowerDeviceStorageAccessInfo-fnptr* args)))

(defonce ^:private LowerIntrin-fnptr* (delay (base/name->global-function "tir.transform.LowerIntrin")))
(defn LowerIntrin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LowerIntrin"}
   (apply base/call-function @LowerIntrin-fnptr* args)))

(defonce ^:private LowerTVMBuiltin-fnptr* (delay (base/name->global-function "tir.transform.LowerTVMBuiltin")))
(defn LowerTVMBuiltin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LowerTVMBuiltin"}
   (apply base/call-function @LowerTVMBuiltin-fnptr* args)))

(defonce ^:private LowerThreadAllreduce-fnptr* (delay (base/name->global-function "tir.transform.LowerThreadAllreduce")))
(defn LowerThreadAllreduce
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LowerThreadAllreduce"}
   (apply base/call-function @LowerThreadAllreduce-fnptr* args)))

(defonce ^:private LowerWarpMemory-fnptr* (delay (base/name->global-function "tir.transform.LowerWarpMemory")))
(defn LowerWarpMemory
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.LowerWarpMemory"}
   (apply base/call-function @LowerWarpMemory-fnptr* args)))

(defonce ^:private MakePackedAPI-fnptr* (delay (base/name->global-function "tir.transform.MakePackedAPI")))
(defn MakePackedAPI
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.MakePackedAPI"}
   (apply base/call-function @MakePackedAPI-fnptr* args)))

(defonce ^:private NarrowDataType-fnptr* (delay (base/name->global-function "tir.transform.NarrowDataType")))
(defn NarrowDataType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.NarrowDataType"}
   (apply base/call-function @NarrowDataType-fnptr* args)))

(defonce ^:private PointerValueTypeRewrite-fnptr* (delay (base/name->global-function "tir.transform.PointerValueTypeRewrite")))
(defn PointerValueTypeRewrite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.PointerValueTypeRewrite"}
   (apply base/call-function @PointerValueTypeRewrite-fnptr* args)))

(defonce ^:private RemapThreadAxis-fnptr* (delay (base/name->global-function "tir.transform.RemapThreadAxis")))
(defn RemapThreadAxis
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.RemapThreadAxis"}
   (apply base/call-function @RemapThreadAxis-fnptr* args)))

(defonce ^:private RemoveNoOp-fnptr* (delay (base/name->global-function "tir.transform.RemoveNoOp")))
(defn RemoveNoOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.RemoveNoOp"}
   (apply base/call-function @RemoveNoOp-fnptr* args)))

(defonce ^:private RewriteUnsafeSelect-fnptr* (delay (base/name->global-function "tir.transform.RewriteUnsafeSelect")))
(defn RewriteUnsafeSelect
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.RewriteUnsafeSelect"}
   (apply base/call-function @RewriteUnsafeSelect-fnptr* args)))

(defonce ^:private Simplify-fnptr* (delay (base/name->global-function "tir.transform.Simplify")))
(defn Simplify
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.Simplify"}
   (apply base/call-function @Simplify-fnptr* args)))

(defonce ^:private SkipAssert-fnptr* (delay (base/name->global-function "tir.transform.SkipAssert")))
(defn SkipAssert
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.SkipAssert"}
   (apply base/call-function @SkipAssert-fnptr* args)))

(defonce ^:private SplitHostDevice-fnptr* (delay (base/name->global-function "tir.transform.SplitHostDevice")))
(defn SplitHostDevice
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.SplitHostDevice"}
   (apply base/call-function @SplitHostDevice-fnptr* args)))

(defonce ^:private StorageFlatten-fnptr* (delay (base/name->global-function "tir.transform.StorageFlatten")))
(defn StorageFlatten
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.StorageFlatten"}
   (apply base/call-function @StorageFlatten-fnptr* args)))

(defonce ^:private StorageRewrite-fnptr* (delay (base/name->global-function "tir.transform.StorageRewrite")))
(defn StorageRewrite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.StorageRewrite"}
   (apply base/call-function @StorageRewrite-fnptr* args)))

(defonce ^:private ThreadSync-fnptr* (delay (base/name->global-function "tir.transform.ThreadSync")))
(defn ThreadSync
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.ThreadSync"}
   (apply base/call-function @ThreadSync-fnptr* args)))

(defonce ^:private UnrollLoop-fnptr* (delay (base/name->global-function "tir.transform.UnrollLoop")))
(defn UnrollLoop
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.UnrollLoop"}
   (apply base/call-function @UnrollLoop-fnptr* args)))

(defonce ^:private VectorizeLoop-fnptr* (delay (base/name->global-function "tir.transform.VectorizeLoop")))
(defn VectorizeLoop
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.VectorizeLoop"}
   (apply base/call-function @VectorizeLoop-fnptr* args)))

(defonce ^:private VerifyGPUCode-fnptr* (delay (base/name->global-function "tir.transform.VerifyGPUCode")))
(defn VerifyGPUCode
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.VerifyGPUCode"}
   (apply base/call-function @VerifyGPUCode-fnptr* args)))

(defonce ^:private VerifyMemory-fnptr* (delay (base/name->global-function "tir.transform.VerifyMemory")))
(defn VerifyMemory
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.VerifyMemory"}
   (apply base/call-function @VerifyMemory-fnptr* args)))

(defonce ^:private VerifySSA-fnptr* (delay (base/name->global-function "tir.transform.VerifySSA")))
(defn VerifySSA
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.transform.VerifySSA"}
   (apply base/call-function @VerifySSA-fnptr* args)))

