(ns tvm-clj.jna.fns.codegen
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LLVMModuleCreate
(let [gfn* (delay (jna-base/name->global-function "codegen.LLVMModuleCreate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} codegen_blob
(let [gfn* (delay (jna-base/name->global-function "codegen.codegen_blob"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} llvm_target_enabled
(let [gfn* (delay (jna-base/name->global-function "codegen.llvm_target_enabled"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

