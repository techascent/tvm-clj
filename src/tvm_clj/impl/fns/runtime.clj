(ns tvm-clj.impl.fns.runtime
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ADT-fnptr* (delay (base/name->global-function "runtime.ADT")))
(defn ADT
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ADT"}
   (apply base/call-function @ADT-fnptr* args)))

(defonce ^:private CSourceModuleCreate-fnptr* (delay (base/name->global-function "runtime.CSourceModuleCreate")))
(defn CSourceModuleCreate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.CSourceModuleCreate"}
   (apply base/call-function @CSourceModuleCreate-fnptr* args)))

(defonce ^:private DumpTypeTable-fnptr* (delay (base/name->global-function "runtime.DumpTypeTable")))
(defn DumpTypeTable
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.DumpTypeTable"}
   (apply base/call-function @DumpTypeTable-fnptr* args)))

(defonce ^:private GetADTFields-fnptr* (delay (base/name->global-function "runtime.GetADTFields")))
(defn GetADTFields
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetADTFields"}
   (apply base/call-function @GetADTFields-fnptr* args)))

(defonce ^:private GetADTSize-fnptr* (delay (base/name->global-function "runtime.GetADTSize")))
(defn GetADTSize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetADTSize"}
   (apply base/call-function @GetADTSize-fnptr* args)))

(defonce ^:private GetADTTag-fnptr* (delay (base/name->global-function "runtime.GetADTTag")))
(defn GetADTTag
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetADTTag"}
   (apply base/call-function @GetADTTag-fnptr* args)))

(defonce ^:private GetDeviceAttr-fnptr* (delay (base/name->global-function "runtime.GetDeviceAttr")))
(defn GetDeviceAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetDeviceAttr"}
   (apply base/call-function @GetDeviceAttr-fnptr* args)))

(defonce ^:private GetFFIString-fnptr* (delay (base/name->global-function "runtime.GetFFIString")))
(defn GetFFIString
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetFFIString"}
   (apply base/call-function @GetFFIString-fnptr* args)))

(defonce ^:private GetGlobalFields-fnptr* (delay (base/name->global-function "runtime.GetGlobalFields")))
(defn GetGlobalFields
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetGlobalFields"}
   (apply base/call-function @GetGlobalFields-fnptr* args)))

(defonce ^:private GetNumOfGlobals-fnptr* (delay (base/name->global-function "runtime.GetNumOfGlobals")))
(defn GetNumOfGlobals
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetNumOfGlobals"}
   (apply base/call-function @GetNumOfGlobals-fnptr* args)))

(defonce ^:private GetNumOfPrimitives-fnptr* (delay (base/name->global-function "runtime.GetNumOfPrimitives")))
(defn GetNumOfPrimitives
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetNumOfPrimitives"}
   (apply base/call-function @GetNumOfPrimitives-fnptr* args)))

(defonce ^:private GetPrimitiveFields-fnptr* (delay (base/name->global-function "runtime.GetPrimitiveFields")))
(defn GetPrimitiveFields
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.GetPrimitiveFields"}
   (apply base/call-function @GetPrimitiveFields-fnptr* args)))

(defonce ^:private Load_Executable-fnptr* (delay (base/name->global-function "runtime.Load_Executable")))
(defn Load_Executable
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.Load_Executable"}
   (apply base/call-function @Load_Executable-fnptr* args)))

(defonce ^:private ModuleGetImport-fnptr* (delay (base/name->global-function "runtime.ModuleGetImport")))
(defn ModuleGetImport
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModuleGetImport"}
   (apply base/call-function @ModuleGetImport-fnptr* args)))

(defonce ^:private ModuleGetSource-fnptr* (delay (base/name->global-function "runtime.ModuleGetSource")))
(defn ModuleGetSource
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModuleGetSource"}
   (apply base/call-function @ModuleGetSource-fnptr* args)))

(defonce ^:private ModuleGetTypeKey-fnptr* (delay (base/name->global-function "runtime.ModuleGetTypeKey")))
(defn ModuleGetTypeKey
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModuleGetTypeKey"}
   (apply base/call-function @ModuleGetTypeKey-fnptr* args)))

(defonce ^:private ModuleImportsSize-fnptr* (delay (base/name->global-function "runtime.ModuleImportsSize")))
(defn ModuleImportsSize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModuleImportsSize"}
   (apply base/call-function @ModuleImportsSize-fnptr* args)))

(defonce ^:private ModuleLoadFromFile-fnptr* (delay (base/name->global-function "runtime.ModuleLoadFromFile")))
(defn ModuleLoadFromFile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModuleLoadFromFile"}
   (apply base/call-function @ModuleLoadFromFile-fnptr* args)))

(defonce ^:private ModulePackImportsToC-fnptr* (delay (base/name->global-function "runtime.ModulePackImportsToC")))
(defn ModulePackImportsToC
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModulePackImportsToC"}
   (apply base/call-function @ModulePackImportsToC-fnptr* args)))

(defonce ^:private ModulePackImportsToLLVM-fnptr* (delay (base/name->global-function "runtime.ModulePackImportsToLLVM")))
(defn ModulePackImportsToLLVM
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModulePackImportsToLLVM"}
   (apply base/call-function @ModulePackImportsToLLVM-fnptr* args)))

(defonce ^:private ModuleSaveToFile-fnptr* (delay (base/name->global-function "runtime.ModuleSaveToFile")))
(defn ModuleSaveToFile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ModuleSaveToFile"}
   (apply base/call-function @ModuleSaveToFile-fnptr* args)))

(defonce ^:private ObjectPtrHash-fnptr* (delay (base/name->global-function "runtime.ObjectPtrHash")))
(defn ObjectPtrHash
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.ObjectPtrHash"}
   (apply base/call-function @ObjectPtrHash-fnptr* args)))

(defonce ^:private RPCTimeEvaluator-fnptr* (delay (base/name->global-function "runtime.RPCTimeEvaluator")))
(defn RPCTimeEvaluator
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.RPCTimeEvaluator"}
   (apply base/call-function @RPCTimeEvaluator-fnptr* args)))

(defonce ^:private RuntimeEnabled-fnptr* (delay (base/name->global-function "runtime.RuntimeEnabled")))
(defn RuntimeEnabled
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.RuntimeEnabled"}
   (apply base/call-function @RuntimeEnabled-fnptr* args)))

(defonce ^:private SourceModuleCreate-fnptr* (delay (base/name->global-function "runtime.SourceModuleCreate")))
(defn SourceModuleCreate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.SourceModuleCreate"}
   (apply base/call-function @SourceModuleCreate-fnptr* args)))

(defonce ^:private String-fnptr* (delay (base/name->global-function "runtime.String")))
(defn RuntimeString
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.String"}
   (apply base/call-function @String-fnptr* args)))

(defonce ^:private SystemLib-fnptr* (delay (base/name->global-function "runtime.SystemLib")))
(defn SystemLib
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.SystemLib"}
   (apply base/call-function @SystemLib-fnptr* args)))

(defonce ^:private TVMSetStream-fnptr* (delay (base/name->global-function "runtime.TVMSetStream")))
(defn TVMSetStream
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.TVMSetStream"}
   (apply base/call-function @TVMSetStream-fnptr* args)))

(defonce ^:private Tuple-fnptr* (delay (base/name->global-function "runtime.Tuple")))
(defn Tuple
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.Tuple"}
   (apply base/call-function @Tuple-fnptr* args)))

(defonce ^:private _VirtualMachine-fnptr* (delay (base/name->global-function "runtime._VirtualMachine")))
(defn _VirtualMachine
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime._VirtualMachine"}
   (apply base/call-function @_VirtualMachine-fnptr* args)))

(defonce ^:private _datatype_get_type_code-fnptr* (delay (base/name->global-function "runtime._datatype_get_type_code")))
(defn _datatype_get_type_code
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime._datatype_get_type_code"}
   (apply base/call-function @_datatype_get_type_code-fnptr* args)))

(defonce ^:private _datatype_get_type_name-fnptr* (delay (base/name->global-function "runtime._datatype_get_type_name")))
(defn _datatype_get_type_name
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime._datatype_get_type_name"}
   (apply base/call-function @_datatype_get_type_name-fnptr* args)))

(defonce ^:private _datatype_get_type_registered-fnptr* (delay (base/name->global-function "runtime._datatype_get_type_registered")))
(defn _datatype_get_type_registered
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime._datatype_get_type_registered"}
   (apply base/call-function @_datatype_get_type_registered-fnptr* args)))

(defonce ^:private _datatype_register-fnptr* (delay (base/name->global-function "runtime._datatype_register")))
(defn _datatype_register
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime._datatype_register"}
   (apply base/call-function @_datatype_register-fnptr* args)))

(defonce ^:private config_threadpool-fnptr* (delay (base/name->global-function "runtime.config_threadpool")))
(defn config_threadpool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.config_threadpool"}
   (apply base/call-function @config_threadpool-fnptr* args)))

