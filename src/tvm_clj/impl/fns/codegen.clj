(ns tvm-clj.impl.fns.codegen
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private LLVMModuleCreate-fnptr* (delay (base/name->global-function "codegen.LLVMModuleCreate")))
(defn LLVMModuleCreate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "codegen.LLVMModuleCreate"}
   (apply base/call-function @LLVMModuleCreate-fnptr* args)))

(defonce ^:private codegen_blob-fnptr* (delay (base/name->global-function "codegen.codegen_blob")))
(defn codegen_blob
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "codegen.codegen_blob"}
   (apply base/call-function @codegen_blob-fnptr* args)))

(defonce ^:private llvm_target_enabled-fnptr* (delay (base/name->global-function "codegen.llvm_target_enabled")))
(defn llvm_target_enabled
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "codegen.llvm_target_enabled"}
   (apply base/call-function @llvm_target_enabled-fnptr* args)))

