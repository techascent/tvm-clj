(ns tvm-clj.jna.fns.runtime
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "runtime.ADT"))]
  (defn ADT
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ADT"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.CSourceModuleCreate"))]
  (defn CSourceModuleCreate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.CSourceModuleCreate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.DumpTypeTable"))]
  (defn DumpTypeTable
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.DumpTypeTable"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetADTFields"))]
  (defn GetADTFields
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetADTFields"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetADTSize"))]
  (defn GetADTSize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetADTSize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetADTTag"))]
  (defn GetADTTag
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetADTTag"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetDeviceAttr"))]
  (defn GetDeviceAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetDeviceAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetFFIString"))]
  (defn GetFFIString
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetFFIString"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetGlobalFields"))]
  (defn GetGlobalFields
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetGlobalFields"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetNumOfGlobals"))]
  (defn GetNumOfGlobals
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetNumOfGlobals"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetNumOfPrimitives"))]
  (defn GetNumOfPrimitives
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetNumOfPrimitives"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.GetPrimitiveFields"))]
  (defn GetPrimitiveFields
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.GetPrimitiveFields"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.Load_Executable"))]
  (defn Load_Executable
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.Load_Executable"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleGetImport"))]
  (defn ModuleGetImport
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModuleGetImport"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleGetSource"))]
  (defn ModuleGetSource
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModuleGetSource"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleGetTypeKey"))]
  (defn ModuleGetTypeKey
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModuleGetTypeKey"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleImportsSize"))]
  (defn ModuleImportsSize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModuleImportsSize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleLoadFromFile"))]
  (defn ModuleLoadFromFile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModuleLoadFromFile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModulePackImportsToC"))]
  (defn ModulePackImportsToC
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModulePackImportsToC"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModulePackImportsToLLVM"))]
  (defn ModulePackImportsToLLVM
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModulePackImportsToLLVM"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleSaveToFile"))]
  (defn ModuleSaveToFile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ModuleSaveToFile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.ObjectPtrHash"))]
  (defn ObjectPtrHash
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.ObjectPtrHash"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.RPCTimeEvaluator"))]
  (defn RPCTimeEvaluator
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.RPCTimeEvaluator"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.RuntimeEnabled"))]
  (defn RuntimeEnabled
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.RuntimeEnabled"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.SourceModuleCreate"))]
  (defn SourceModuleCreate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.SourceModuleCreate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.String"))]
  (defn RuntimeString
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.String"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.SystemLib"))]
  (defn SystemLib
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.SystemLib"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.TVMSetStream"))]
  (defn TVMSetStream
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.TVMSetStream"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.Tuple"))]
  (defn Tuple
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.Tuple"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime._VirtualMachine"))]
  (defn _VirtualMachine
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime._VirtualMachine"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_get_type_code"))]
  (defn _datatype_get_type_code
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime._datatype_get_type_code"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_get_type_name"))]
  (defn _datatype_get_type_name
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime._datatype_get_type_name"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_get_type_registered"))]
  (defn _datatype_get_type_registered
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime._datatype_get_type_registered"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_register"))]
  (defn _datatype_register
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime._datatype_register"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.config_threadpool"))]
  (defn config_threadpool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.config_threadpool"}
     (apply jna-base/call-function @gfn* args))))

