(ns tvm-clj.jna.fns.runtime
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ADT
(let [gfn* (delay (jna-base/name->global-function "runtime.ADT"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CSourceModuleCreate
(let [gfn* (delay (jna-base/name->global-function "runtime.CSourceModuleCreate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DumpTypeTable
(let [gfn* (delay (jna-base/name->global-function "runtime.DumpTypeTable"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetADTFields
(let [gfn* (delay (jna-base/name->global-function "runtime.GetADTFields"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetADTSize
(let [gfn* (delay (jna-base/name->global-function "runtime.GetADTSize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetADTTag
(let [gfn* (delay (jna-base/name->global-function "runtime.GetADTTag"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetDeviceAttr
(let [gfn* (delay (jna-base/name->global-function "runtime.GetDeviceAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetFFIString
(let [gfn* (delay (jna-base/name->global-function "runtime.GetFFIString"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetGlobalFields
(let [gfn* (delay (jna-base/name->global-function "runtime.GetGlobalFields"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetNumOfGlobals
(let [gfn* (delay (jna-base/name->global-function "runtime.GetNumOfGlobals"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetNumOfPrimitives
(let [gfn* (delay (jna-base/name->global-function "runtime.GetNumOfPrimitives"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetPrimitiveFields
(let [gfn* (delay (jna-base/name->global-function "runtime.GetPrimitiveFields"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Load_Executable
(let [gfn* (delay (jna-base/name->global-function "runtime.Load_Executable"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleGetImport
(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleGetImport"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleGetSource
(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleGetSource"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleGetTypeKey
(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleGetTypeKey"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleImportsSize
(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleImportsSize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleLoadFromFile
(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleLoadFromFile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModulePackImportsToC
(let [gfn* (delay (jna-base/name->global-function "runtime.ModulePackImportsToC"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModulePackImportsToLLVM
(let [gfn* (delay (jna-base/name->global-function "runtime.ModulePackImportsToLLVM"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleSaveToFile
(let [gfn* (delay (jna-base/name->global-function "runtime.ModuleSaveToFile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ObjectPtrHash
(let [gfn* (delay (jna-base/name->global-function "runtime.ObjectPtrHash"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RPCTimeEvaluator
(let [gfn* (delay (jna-base/name->global-function "runtime.RPCTimeEvaluator"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RuntimeEnabled
(let [gfn* (delay (jna-base/name->global-function "runtime.RuntimeEnabled"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SourceModuleCreate
(let [gfn* (delay (jna-base/name->global-function "runtime.SourceModuleCreate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} String
(let [gfn* (delay (jna-base/name->global-function "runtime.String"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SystemLib
(let [gfn* (delay (jna-base/name->global-function "runtime.SystemLib"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TVMSetStream
(let [gfn* (delay (jna-base/name->global-function "runtime.TVMSetStream"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Tuple
(let [gfn* (delay (jna-base/name->global-function "runtime.Tuple"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _VirtualMachine
(let [gfn* (delay (jna-base/name->global-function "runtime._VirtualMachine"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _datatype_get_type_code
(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_get_type_code"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _datatype_get_type_name
(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_get_type_name"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _datatype_get_type_registered
(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_get_type_registered"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _datatype_register
(let [gfn* (delay (jna-base/name->global-function "runtime._datatype_register"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} config_threadpool
(let [gfn* (delay (jna-base/name->global-function "runtime.config_threadpool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

