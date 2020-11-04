(ns tvm-clj.jna.fns.ir
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "ir.AsText"))]
  (defn AsText
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.AsText"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.AttrsListFieldInfo"))]
  (defn AttrsListFieldInfo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.AttrsListFieldInfo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.BaseFuncCopy"))]
  (defn BaseFuncCopy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.BaseFuncCopy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.BaseFuncWithAttr"))]
  (defn BaseFuncWithAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.BaseFuncWithAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.BaseFunc_Attrs"))]
  (defn BaseFunc_Attrs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.BaseFunc_Attrs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Constructor"))]
  (defn Constructor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Constructor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.DebugPrint"))]
  (defn DebugPrint
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.DebugPrint"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.DictAttrsGetDict"))]
  (defn DictAttrsGetDict
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.DictAttrsGetDict"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.EnvFuncCall"))]
  (defn EnvFuncCall
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.EnvFuncCall"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.EnvFuncGet"))]
  (defn EnvFuncGet
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.EnvFuncGet"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.EnvFuncGetPackedFunc"))]
  (defn EnvFuncGetPackedFunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.EnvFuncGetPackedFunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.FloatImm"))]
  (defn FloatImm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.FloatImm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.FuncType"))]
  (defn FuncType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.FuncType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.GetOp"))]
  (defn GetOp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.GetOp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.GlobalTypeVar"))]
  (defn GlobalTypeVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.GlobalTypeVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.GlobalVar"))]
  (defn GlobalVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.GlobalVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.IRModule"))]
  (defn IRModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.IRModule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.IncompleteType"))]
  (defn IncompleteType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.IncompleteType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.IntImm"))]
  (defn IntImm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.IntImm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.ListOpNames"))]
  (defn ListOpNames
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.ListOpNames"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_Add"))]
  (defn Module_Add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_Add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_AddDef"))]
  (defn Module_AddDef
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_AddDef"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_ContainGlobalVar"))]
  (defn Module_ContainGlobalVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_ContainGlobalVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_FromExpr"))]
  (defn Module_FromExpr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_FromExpr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalTypeVar"))]
  (defn Module_GetGlobalTypeVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_GetGlobalTypeVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalTypeVars"))]
  (defn Module_GetGlobalTypeVars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_GetGlobalTypeVars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalVar"))]
  (defn Module_GetGlobalVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_GetGlobalVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_GetGlobalVars"))]
  (defn Module_GetGlobalVars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_GetGlobalVars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_Import"))]
  (defn Module_Import
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_Import"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_ImportFromStd"))]
  (defn Module_ImportFromStd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_ImportFromStd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_Lookup"))]
  (defn Module_Lookup
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_Lookup"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_LookupDef"))]
  (defn Module_LookupDef
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_LookupDef"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_LookupDef_str"))]
  (defn Module_LookupDef_str
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_LookupDef_str"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_LookupTag"))]
  (defn Module_LookupTag
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_LookupTag"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_Lookup_str"))]
  (defn Module_Lookup_str
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_Lookup_str"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_Update"))]
  (defn Module_Update
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_Update"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Module_UpdateFunction"))]
  (defn Module_UpdateFunction
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Module_UpdateFunction"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.NodeSetSpan"))]
  (defn NodeSetSpan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.NodeSetSpan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.OpGetAttr"))]
  (defn OpGetAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.OpGetAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.OpResetAttr"))]
  (defn OpResetAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.OpResetAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.OpSetAttr"))]
  (defn OpSetAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.OpSetAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.PointerType"))]
  (defn PointerType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.PointerType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.PrettyPrint"))]
  (defn PrettyPrint
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.PrettyPrint"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.PrimType"))]
  (defn PrimType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.PrimType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Range"))]
  (defn Range
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Range"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Range_from_min_extent"))]
  (defn Range_from_min_extent
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Range_from_min_extent"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.RegisterOpAttr"))]
  (defn RegisterOpAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.RegisterOpAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.RelayRefType"))]
  (defn RelayRefType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.RelayRefType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.SourceName"))]
  (defn SourceName
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.SourceName"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.Span"))]
  (defn Span
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.Span"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TensorType"))]
  (defn TensorType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TensorType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TextPrinter"))]
  (defn TextPrinter
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TextPrinter"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TupleType"))]
  (defn TupleType
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TupleType"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TypeCall"))]
  (defn TypeCall
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TypeCall"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TypeData"))]
  (defn TypeData
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TypeData"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TypeRelation"))]
  (defn TypeRelation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TypeRelation"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "ir.TypeVar"))]
  (defn TypeVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "ir.TypeVar"}
     (apply jna-base/call-function @gfn* args))))

