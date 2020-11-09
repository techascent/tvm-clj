(ns tvm-clj.impl.fns.tvm.codegen.llvm
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private target_arm-fnptr* (delay (base/name->global-function "tvm.codegen.llvm.target_arm")))
(defn target_arm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.codegen.llvm.target_arm"}
   (apply base/call-function @target_arm-fnptr* args)))

(defonce ^:private target_x86-64-fnptr* (delay (base/name->global-function "tvm.codegen.llvm.target_x86-64")))
(defn target_x86-64
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.codegen.llvm.target_x86-64"}
   (apply base/call-function @target_x86-64-fnptr* args)))

