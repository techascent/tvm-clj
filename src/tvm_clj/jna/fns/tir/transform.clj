(ns tvm-clj.jna.fns.tir.transform
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BF16CastElimination
(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16CastElimination"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BF16Legalize
(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16Legalize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BF16Promote
(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16Promote"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BF16TypeLowering
(let [gfn* (delay (jna-base/name->global-function "tir.transform.BF16TypeLowering"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CoProcSync
(let [gfn* (delay (jna-base/name->global-function "tir.transform.CoProcSync"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CombineContextCall
(let [gfn* (delay (jna-base/name->global-function "tir.transform.CombineContextCall"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreatePrimFuncPass
(let [gfn* (delay (jna-base/name->global-function "tir.transform.CreatePrimFuncPass"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DecorateDeviceScope
(let [gfn* (delay (jna-base/name->global-function "tir.transform.DecorateDeviceScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} HoistIfThenElse
(let [gfn* (delay (jna-base/name->global-function "tir.transform.HoistIfThenElse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} HoistIfThenElseBasic
(let [gfn* (delay (jna-base/name->global-function "tir.transform.HoistIfThenElseBasic"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InferFragment
(let [gfn* (delay (jna-base/name->global-function "tir.transform.InferFragment"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InjectCopyIntrin
(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectCopyIntrin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InjectDoubleBuffer
(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectDoubleBuffer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InjectPrefetch
(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectPrefetch"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InjectVirtualThread
(let [gfn* (delay (jna-base/name->global-function "tir.transform.InjectVirtualThread"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InstrumentBoundCheckers
(let [gfn* (delay (jna-base/name->global-function "tir.transform.InstrumentBoundCheckers"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LiftAttrScope
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LiftAttrScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LoopPartition
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LoopPartition"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LowerCustomDatatypes
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerCustomDatatypes"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LowerDeviceStorageAccessInfo
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerDeviceStorageAccessInfo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LowerIntrin
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerIntrin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LowerTVMBuiltin
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerTVMBuiltin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LowerThreadAllreduce
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerThreadAllreduce"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LowerWarpMemory
(let [gfn* (delay (jna-base/name->global-function "tir.transform.LowerWarpMemory"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MakePackedAPI
(let [gfn* (delay (jna-base/name->global-function "tir.transform.MakePackedAPI"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NarrowDataType
(let [gfn* (delay (jna-base/name->global-function "tir.transform.NarrowDataType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PointerValueTypeRewrite
(let [gfn* (delay (jna-base/name->global-function "tir.transform.PointerValueTypeRewrite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RemapThreadAxis
(let [gfn* (delay (jna-base/name->global-function "tir.transform.RemapThreadAxis"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RemoveNoOp
(let [gfn* (delay (jna-base/name->global-function "tir.transform.RemoveNoOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RewriteUnsafeSelect
(let [gfn* (delay (jna-base/name->global-function "tir.transform.RewriteUnsafeSelect"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Simplify
(let [gfn* (delay (jna-base/name->global-function "tir.transform.Simplify"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SkipAssert
(let [gfn* (delay (jna-base/name->global-function "tir.transform.SkipAssert"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SplitHostDevice
(let [gfn* (delay (jna-base/name->global-function "tir.transform.SplitHostDevice"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StorageFlatten
(let [gfn* (delay (jna-base/name->global-function "tir.transform.StorageFlatten"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StorageRewrite
(let [gfn* (delay (jna-base/name->global-function "tir.transform.StorageRewrite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ThreadSync
(let [gfn* (delay (jna-base/name->global-function "tir.transform.ThreadSync"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} UnrollLoop
(let [gfn* (delay (jna-base/name->global-function "tir.transform.UnrollLoop"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} VectorizeLoop
(let [gfn* (delay (jna-base/name->global-function "tir.transform.VectorizeLoop"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} VerifyGPUCode
(let [gfn* (delay (jna-base/name->global-function "tir.transform.VerifyGPUCode"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} VerifyMemory
(let [gfn* (delay (jna-base/name->global-function "tir.transform.VerifyMemory"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} VerifySSA
(let [gfn* (delay (jna-base/name->global-function "tir.transform.VerifySSA"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

