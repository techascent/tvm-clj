(ns tvm-clj.impl.fns.ir
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AsText-fnptr* (delay (base/name->global-function "ir.AsText")))
(defn AsText
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.AsText"}
   (apply base/call-function @AsText-fnptr* args)))

(defonce ^:private AttrsListFieldInfo-fnptr* (delay (base/name->global-function "ir.AttrsListFieldInfo")))
(defn AttrsListFieldInfo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.AttrsListFieldInfo"}
   (apply base/call-function @AttrsListFieldInfo-fnptr* args)))

(defonce ^:private BaseFuncCopy-fnptr* (delay (base/name->global-function "ir.BaseFuncCopy")))
(defn BaseFuncCopy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.BaseFuncCopy"}
   (apply base/call-function @BaseFuncCopy-fnptr* args)))

(defonce ^:private BaseFuncWithAttr-fnptr* (delay (base/name->global-function "ir.BaseFuncWithAttr")))
(defn BaseFuncWithAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.BaseFuncWithAttr"}
   (apply base/call-function @BaseFuncWithAttr-fnptr* args)))

(defonce ^:private BaseFunc_Attrs-fnptr* (delay (base/name->global-function "ir.BaseFunc_Attrs")))
(defn BaseFunc_Attrs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.BaseFunc_Attrs"}
   (apply base/call-function @BaseFunc_Attrs-fnptr* args)))

(defonce ^:private Constructor-fnptr* (delay (base/name->global-function "ir.Constructor")))
(defn Constructor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Constructor"}
   (apply base/call-function @Constructor-fnptr* args)))

(defonce ^:private DebugPrint-fnptr* (delay (base/name->global-function "ir.DebugPrint")))
(defn DebugPrint
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.DebugPrint"}
   (apply base/call-function @DebugPrint-fnptr* args)))

(defonce ^:private DictAttrsGetDict-fnptr* (delay (base/name->global-function "ir.DictAttrsGetDict")))
(defn DictAttrsGetDict
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.DictAttrsGetDict"}
   (apply base/call-function @DictAttrsGetDict-fnptr* args)))

(defonce ^:private EnvFuncCall-fnptr* (delay (base/name->global-function "ir.EnvFuncCall")))
(defn EnvFuncCall
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.EnvFuncCall"}
   (apply base/call-function @EnvFuncCall-fnptr* args)))

(defonce ^:private EnvFuncGet-fnptr* (delay (base/name->global-function "ir.EnvFuncGet")))
(defn EnvFuncGet
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.EnvFuncGet"}
   (apply base/call-function @EnvFuncGet-fnptr* args)))

(defonce ^:private EnvFuncGetPackedFunc-fnptr* (delay (base/name->global-function "ir.EnvFuncGetPackedFunc")))
(defn EnvFuncGetPackedFunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.EnvFuncGetPackedFunc"}
   (apply base/call-function @EnvFuncGetPackedFunc-fnptr* args)))

(defonce ^:private FloatImm-fnptr* (delay (base/name->global-function "ir.FloatImm")))
(defn FloatImm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.FloatImm"}
   (apply base/call-function @FloatImm-fnptr* args)))

(defonce ^:private FuncType-fnptr* (delay (base/name->global-function "ir.FuncType")))
(defn FuncType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.FuncType"}
   (apply base/call-function @FuncType-fnptr* args)))

(defonce ^:private GetOp-fnptr* (delay (base/name->global-function "ir.GetOp")))
(defn GetOp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.GetOp"}
   (apply base/call-function @GetOp-fnptr* args)))

(defonce ^:private GlobalTypeVar-fnptr* (delay (base/name->global-function "ir.GlobalTypeVar")))
(defn GlobalTypeVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.GlobalTypeVar"}
   (apply base/call-function @GlobalTypeVar-fnptr* args)))

(defonce ^:private GlobalVar-fnptr* (delay (base/name->global-function "ir.GlobalVar")))
(defn GlobalVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.GlobalVar"}
   (apply base/call-function @GlobalVar-fnptr* args)))

(defonce ^:private IRModule-fnptr* (delay (base/name->global-function "ir.IRModule")))
(defn IRModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.IRModule"}
   (apply base/call-function @IRModule-fnptr* args)))

(defonce ^:private IncompleteType-fnptr* (delay (base/name->global-function "ir.IncompleteType")))
(defn IncompleteType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.IncompleteType"}
   (apply base/call-function @IncompleteType-fnptr* args)))

(defonce ^:private IntImm-fnptr* (delay (base/name->global-function "ir.IntImm")))
(defn IntImm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.IntImm"}
   (apply base/call-function @IntImm-fnptr* args)))

(defonce ^:private ListOpNames-fnptr* (delay (base/name->global-function "ir.ListOpNames")))
(defn ListOpNames
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.ListOpNames"}
   (apply base/call-function @ListOpNames-fnptr* args)))

(defonce ^:private Module_Add-fnptr* (delay (base/name->global-function "ir.Module_Add")))
(defn Module_Add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_Add"}
   (apply base/call-function @Module_Add-fnptr* args)))

(defonce ^:private Module_AddDef-fnptr* (delay (base/name->global-function "ir.Module_AddDef")))
(defn Module_AddDef
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_AddDef"}
   (apply base/call-function @Module_AddDef-fnptr* args)))

(defonce ^:private Module_ContainGlobalVar-fnptr* (delay (base/name->global-function "ir.Module_ContainGlobalVar")))
(defn Module_ContainGlobalVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_ContainGlobalVar"}
   (apply base/call-function @Module_ContainGlobalVar-fnptr* args)))

(defonce ^:private Module_FromExpr-fnptr* (delay (base/name->global-function "ir.Module_FromExpr")))
(defn Module_FromExpr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_FromExpr"}
   (apply base/call-function @Module_FromExpr-fnptr* args)))

(defonce ^:private Module_GetGlobalTypeVar-fnptr* (delay (base/name->global-function "ir.Module_GetGlobalTypeVar")))
(defn Module_GetGlobalTypeVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_GetGlobalTypeVar"}
   (apply base/call-function @Module_GetGlobalTypeVar-fnptr* args)))

(defonce ^:private Module_GetGlobalTypeVars-fnptr* (delay (base/name->global-function "ir.Module_GetGlobalTypeVars")))
(defn Module_GetGlobalTypeVars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_GetGlobalTypeVars"}
   (apply base/call-function @Module_GetGlobalTypeVars-fnptr* args)))

(defonce ^:private Module_GetGlobalVar-fnptr* (delay (base/name->global-function "ir.Module_GetGlobalVar")))
(defn Module_GetGlobalVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_GetGlobalVar"}
   (apply base/call-function @Module_GetGlobalVar-fnptr* args)))

(defonce ^:private Module_GetGlobalVars-fnptr* (delay (base/name->global-function "ir.Module_GetGlobalVars")))
(defn Module_GetGlobalVars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_GetGlobalVars"}
   (apply base/call-function @Module_GetGlobalVars-fnptr* args)))

(defonce ^:private Module_Import-fnptr* (delay (base/name->global-function "ir.Module_Import")))
(defn Module_Import
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_Import"}
   (apply base/call-function @Module_Import-fnptr* args)))

(defonce ^:private Module_ImportFromStd-fnptr* (delay (base/name->global-function "ir.Module_ImportFromStd")))
(defn Module_ImportFromStd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_ImportFromStd"}
   (apply base/call-function @Module_ImportFromStd-fnptr* args)))

(defonce ^:private Module_Lookup-fnptr* (delay (base/name->global-function "ir.Module_Lookup")))
(defn Module_Lookup
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_Lookup"}
   (apply base/call-function @Module_Lookup-fnptr* args)))

(defonce ^:private Module_LookupDef-fnptr* (delay (base/name->global-function "ir.Module_LookupDef")))
(defn Module_LookupDef
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_LookupDef"}
   (apply base/call-function @Module_LookupDef-fnptr* args)))

(defonce ^:private Module_LookupDef_str-fnptr* (delay (base/name->global-function "ir.Module_LookupDef_str")))
(defn Module_LookupDef_str
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_LookupDef_str"}
   (apply base/call-function @Module_LookupDef_str-fnptr* args)))

(defonce ^:private Module_LookupTag-fnptr* (delay (base/name->global-function "ir.Module_LookupTag")))
(defn Module_LookupTag
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_LookupTag"}
   (apply base/call-function @Module_LookupTag-fnptr* args)))

(defonce ^:private Module_Lookup_str-fnptr* (delay (base/name->global-function "ir.Module_Lookup_str")))
(defn Module_Lookup_str
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_Lookup_str"}
   (apply base/call-function @Module_Lookup_str-fnptr* args)))

(defonce ^:private Module_Update-fnptr* (delay (base/name->global-function "ir.Module_Update")))
(defn Module_Update
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_Update"}
   (apply base/call-function @Module_Update-fnptr* args)))

(defonce ^:private Module_UpdateFunction-fnptr* (delay (base/name->global-function "ir.Module_UpdateFunction")))
(defn Module_UpdateFunction
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Module_UpdateFunction"}
   (apply base/call-function @Module_UpdateFunction-fnptr* args)))

(defonce ^:private NodeSetSpan-fnptr* (delay (base/name->global-function "ir.NodeSetSpan")))
(defn NodeSetSpan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.NodeSetSpan"}
   (apply base/call-function @NodeSetSpan-fnptr* args)))

(defonce ^:private OpGetAttr-fnptr* (delay (base/name->global-function "ir.OpGetAttr")))
(defn OpGetAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.OpGetAttr"}
   (apply base/call-function @OpGetAttr-fnptr* args)))

(defonce ^:private OpResetAttr-fnptr* (delay (base/name->global-function "ir.OpResetAttr")))
(defn OpResetAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.OpResetAttr"}
   (apply base/call-function @OpResetAttr-fnptr* args)))

(defonce ^:private OpSetAttr-fnptr* (delay (base/name->global-function "ir.OpSetAttr")))
(defn OpSetAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.OpSetAttr"}
   (apply base/call-function @OpSetAttr-fnptr* args)))

(defonce ^:private PointerType-fnptr* (delay (base/name->global-function "ir.PointerType")))
(defn PointerType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.PointerType"}
   (apply base/call-function @PointerType-fnptr* args)))

(defonce ^:private PrettyPrint-fnptr* (delay (base/name->global-function "ir.PrettyPrint")))
(defn PrettyPrint
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.PrettyPrint"}
   (apply base/call-function @PrettyPrint-fnptr* args)))

(defonce ^:private PrimType-fnptr* (delay (base/name->global-function "ir.PrimType")))
(defn PrimType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.PrimType"}
   (apply base/call-function @PrimType-fnptr* args)))

(defonce ^:private Range-fnptr* (delay (base/name->global-function "ir.Range")))
(defn Range
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Range"}
   (apply base/call-function @Range-fnptr* args)))

(defonce ^:private Range_from_min_extent-fnptr* (delay (base/name->global-function "ir.Range_from_min_extent")))
(defn Range_from_min_extent
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Range_from_min_extent"}
   (apply base/call-function @Range_from_min_extent-fnptr* args)))

(defonce ^:private RegisterOpAttr-fnptr* (delay (base/name->global-function "ir.RegisterOpAttr")))
(defn RegisterOpAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.RegisterOpAttr"}
   (apply base/call-function @RegisterOpAttr-fnptr* args)))

(defonce ^:private RelayRefType-fnptr* (delay (base/name->global-function "ir.RelayRefType")))
(defn RelayRefType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.RelayRefType"}
   (apply base/call-function @RelayRefType-fnptr* args)))

(defonce ^:private SourceName-fnptr* (delay (base/name->global-function "ir.SourceName")))
(defn SourceName
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.SourceName"}
   (apply base/call-function @SourceName-fnptr* args)))

(defonce ^:private Span-fnptr* (delay (base/name->global-function "ir.Span")))
(defn Span
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.Span"}
   (apply base/call-function @Span-fnptr* args)))

(defonce ^:private TensorType-fnptr* (delay (base/name->global-function "ir.TensorType")))
(defn TensorType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TensorType"}
   (apply base/call-function @TensorType-fnptr* args)))

(defonce ^:private TextPrinter-fnptr* (delay (base/name->global-function "ir.TextPrinter")))
(defn TextPrinter
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TextPrinter"}
   (apply base/call-function @TextPrinter-fnptr* args)))

(defonce ^:private TupleType-fnptr* (delay (base/name->global-function "ir.TupleType")))
(defn TupleType
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TupleType"}
   (apply base/call-function @TupleType-fnptr* args)))

(defonce ^:private TypeCall-fnptr* (delay (base/name->global-function "ir.TypeCall")))
(defn TypeCall
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TypeCall"}
   (apply base/call-function @TypeCall-fnptr* args)))

(defonce ^:private TypeData-fnptr* (delay (base/name->global-function "ir.TypeData")))
(defn TypeData
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TypeData"}
   (apply base/call-function @TypeData-fnptr* args)))

(defonce ^:private TypeRelation-fnptr* (delay (base/name->global-function "ir.TypeRelation")))
(defn TypeRelation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TypeRelation"}
   (apply base/call-function @TypeRelation-fnptr* args)))

(defonce ^:private TypeVar-fnptr* (delay (base/name->global-function "ir.TypeVar")))
(defn TypeVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "ir.TypeVar"}
   (apply base/call-function @TypeVar-fnptr* args)))

