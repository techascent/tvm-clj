(ns tvm-clj.jna.fns.tvm.codegen.llvm
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.codegen.llvm.target_arm"))]
  (defn target_arm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.codegen.llvm.target_arm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.codegen.llvm.target_x86-64"))]
  (defn target_x86-64
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.codegen.llvm.target_x86-64"}
     (apply jna-base/call-function @gfn* args))))

