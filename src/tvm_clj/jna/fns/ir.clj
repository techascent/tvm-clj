(ns tvm-clj.jna.fns.ir
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AsText
(let [gfn* (delay (jna-base/name->global-function "ir.AsText"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AttrsListFieldInfo
(let [gfn* (delay (jna-base/name->global-function "ir.AttrsListFieldInfo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BaseFuncCopy
(let [gfn* (delay (jna-base/name->global-function "ir.BaseFuncCopy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BaseFuncWithAttr
(let [gfn* (delay (jna-base/name->global-function "ir.BaseFuncWithAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BaseFunc_Attrs
(let [gfn* (delay (jna-base/name->global-function "ir.BaseFunc_Attrs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Constructor
(let [gfn* (delay (jna-base/name->global-function "ir.Constructor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DebugPrint
(let [gfn* (delay (jna-base/name->global-function "ir.DebugPrint"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DictAttrsGetDict
(let [gfn* (delay (jna-base/name->global-function "ir.DictAttrsGetDict"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EnvFuncCall
(let [gfn* (delay (jna-base/name->global-function "ir.EnvFuncCall"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EnvFuncGet
(let [gfn* (delay (jna-base/name->global-function "ir.EnvFuncGet"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EnvFuncGetPackedFunc
(let [gfn* (delay (jna-base/name->global-function "ir.EnvFuncGetPackedFunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FloatImm
(let [gfn* (delay (jna-base/name->global-function "ir.FloatImm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FuncType
(let [gfn* (delay (jna-base/name->global-function "ir.FuncType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetOp
(let [gfn* (delay (jna-base/name->global-function "ir.GetOp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GlobalTypeVar
(let [gfn* (delay (jna-base/name->global-function "ir.GlobalTypeVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GlobalVar
(let [gfn* (delay (jna-base/name->global-function "ir.GlobalVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IRModule
(let [gfn* (delay (jna-base/name->global-function "ir.IRModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IncompleteType
(let [gfn* (delay (jna-base/name->global-function "ir.IncompleteType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntImm
(let [gfn* (delay (jna-base/name->global-function "ir.IntImm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ListOpNames
(let [gfn* (delay (jna-base/name->global-function "ir.ListOpNames"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_Add
(let [gfn* (delay (jna-base/name->global-function "ir.Module_Add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_AddDef
(let [gfn* (delay (jna-base/name->global-function "ir.Module_AddDef"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_ContainGlobalVar
(let [gfn* (delay (jna-base/name->global-function "ir.Module_ContainGlobalVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_FromExpr
(let [gfn* (delay (jna-base/name->global-function "ir.Module_FromExpr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_GetGlobalTypeVar
(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalTypeVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_GetGlobalTypeVars
(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalTypeVars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_GetGlobalVar
(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_GetGlobalVars
(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalVars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_Import
(let [gfn* (delay (jna-base/name->global-function "ir.Module_Import"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_ImportFromStd
(let [gfn* (delay (jna-base/name->global-function "ir.Module_ImportFromStd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_Lookup
(let [gfn* (delay (jna-base/name->global-function "ir.Module_Lookup"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_LookupDef
(let [gfn* (delay (jna-base/name->global-function "ir.Module_LookupDef"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_LookupDef_str
(let [gfn* (delay (jna-base/name->global-function "ir.Module_LookupDef_str"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_LookupTag
(let [gfn* (delay (jna-base/name->global-function "ir.Module_LookupTag"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_Lookup_str
(let [gfn* (delay (jna-base/name->global-function "ir.Module_Lookup_str"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_Update
(let [gfn* (delay (jna-base/name->global-function "ir.Module_Update"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Module_UpdateFunction
(let [gfn* (delay (jna-base/name->global-function "ir.Module_UpdateFunction"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NodeSetSpan
(let [gfn* (delay (jna-base/name->global-function "ir.NodeSetSpan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpGetAttr
(let [gfn* (delay (jna-base/name->global-function "ir.OpGetAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpResetAttr
(let [gfn* (delay (jna-base/name->global-function "ir.OpResetAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpSetAttr
(let [gfn* (delay (jna-base/name->global-function "ir.OpSetAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PointerType
(let [gfn* (delay (jna-base/name->global-function "ir.PointerType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PrettyPrint
(let [gfn* (delay (jna-base/name->global-function "ir.PrettyPrint"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PrimType
(let [gfn* (delay (jna-base/name->global-function "ir.PrimType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Range
(let [gfn* (delay (jna-base/name->global-function "ir.Range"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Range_from_min_extent
(let [gfn* (delay (jna-base/name->global-function "ir.Range_from_min_extent"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RegisterOpAttr
(let [gfn* (delay (jna-base/name->global-function "ir.RegisterOpAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RelayRefType
(let [gfn* (delay (jna-base/name->global-function "ir.RelayRefType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SourceName
(let [gfn* (delay (jna-base/name->global-function "ir.SourceName"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Span
(let [gfn* (delay (jna-base/name->global-function "ir.Span"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TensorType
(let [gfn* (delay (jna-base/name->global-function "ir.TensorType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TextPrinter
(let [gfn* (delay (jna-base/name->global-function "ir.TextPrinter"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TupleType
(let [gfn* (delay (jna-base/name->global-function "ir.TupleType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TypeCall
(let [gfn* (delay (jna-base/name->global-function "ir.TypeCall"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TypeData
(let [gfn* (delay (jna-base/name->global-function "ir.TypeData"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TypeRelation
(let [gfn* (delay (jna-base/name->global-function "ir.TypeRelation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TypeVar
(let [gfn* (delay (jna-base/name->global-function "ir.TypeVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

