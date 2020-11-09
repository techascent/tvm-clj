(ns tvm-clj.jna.fns.tir.transform
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16CastElimination"))]
  (defn BF16CastElimination
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.BF16CastElimination"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16Legalize"))]
  (defn BF16Legalize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.BF16Legalize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16Promote"))]
  (defn BF16Promote
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.BF16Promote"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16TypeLowering"))]
  (defn BF16TypeLowering
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.BF16TypeLowering"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.CoProcSync"))]
  (defn CoProcSync
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.CoProcSync"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.CombineContextCall"))]
  (defn CombineContextCall
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.CombineContextCall"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.CreatePrimFuncPass"))]
  (defn CreatePrimFuncPass
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.CreatePrimFuncPass"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.DecorateDeviceScope"))]
  (defn DecorateDeviceScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.DecorateDeviceScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.HoistIfThenElse"))]
  (defn HoistIfThenElse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.HoistIfThenElse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.HoistIfThenElseBasic"))]
  (defn HoistIfThenElseBasic
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.HoistIfThenElseBasic"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.InferFragment"))]
  (defn InferFragment
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.InferFragment"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectCopyIntrin"))]
  (defn InjectCopyIntrin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.InjectCopyIntrin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectDoubleBuffer"))]
  (defn InjectDoubleBuffer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.InjectDoubleBuffer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectPrefetch"))]
  (defn InjectPrefetch
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.InjectPrefetch"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectVirtualThread"))]
  (defn InjectVirtualThread
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.InjectVirtualThread"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.InstrumentBoundCheckers"))]
  (defn InstrumentBoundCheckers
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.InstrumentBoundCheckers"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LiftAttrScope"))]
  (defn LiftAttrScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LiftAttrScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LoopPartition"))]
  (defn LoopPartition
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LoopPartition"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerCustomDatatypes"))]
  (defn LowerCustomDatatypes
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LowerCustomDatatypes"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerDeviceStorageAccessInfo"))]
  (defn LowerDeviceStorageAccessInfo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LowerDeviceStorageAccessInfo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerIntrin"))]
  (defn LowerIntrin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LowerIntrin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerTVMBuiltin"))]
  (defn LowerTVMBuiltin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LowerTVMBuiltin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerThreadAllreduce"))]
  (defn LowerThreadAllreduce
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LowerThreadAllreduce"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerWarpMemory"))]
  (defn LowerWarpMemory
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.LowerWarpMemory"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.MakePackedAPI"))]
  (defn MakePackedAPI
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.MakePackedAPI"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.NarrowDataType"))]
  (defn NarrowDataType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.NarrowDataType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.PointerValueTypeRewrite"))]
  (defn PointerValueTypeRewrite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.PointerValueTypeRewrite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.RemapThreadAxis"))]
  (defn RemapThreadAxis
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.RemapThreadAxis"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.RemoveNoOp"))]
  (defn RemoveNoOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.RemoveNoOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.RewriteUnsafeSelect"))]
  (defn RewriteUnsafeSelect
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.RewriteUnsafeSelect"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.Simplify"))]
  (defn Simplify
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.Simplify"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.SkipAssert"))]
  (defn SkipAssert
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.SkipAssert"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.SplitHostDevice"))]
  (defn SplitHostDevice
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.SplitHostDevice"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.StorageFlatten"))]
  (defn StorageFlatten
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.StorageFlatten"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.StorageRewrite"))]
  (defn StorageRewrite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.StorageRewrite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.ThreadSync"))]
  (defn ThreadSync
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.ThreadSync"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.UnrollLoop"))]
  (defn UnrollLoop
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.UnrollLoop"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.VectorizeLoop"))]
  (defn VectorizeLoop
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.VectorizeLoop"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.VerifyGPUCode"))]
  (defn VerifyGPUCode
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.VerifyGPUCode"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.VerifyMemory"))]
  (defn VerifyMemory
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.VerifyMemory"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.transform.VerifySSA"))]
  (defn VerifySSA
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.transform.VerifySSA"}
     (apply jna-base/call-function @gfn* args))))

