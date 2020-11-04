(ns tvm-clj.jna.fns.codegen
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "codegen.LLVMModuleCreate"))]
  (defn LLVMModuleCreate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "codegen.LLVMModuleCreate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "codegen.codegen_blob"))]
  (defn codegen_blob
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "codegen.codegen_blob"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "codegen.llvm_target_enabled"))]
  (defn llvm_target_enabled
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "codegen.llvm_target_enabled"}
     (apply jna-base/call-function @gfn* args))))

