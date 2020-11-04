(ns tvm-clj.jna.fns.tvm.codegen.llvm
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} target_arm
(let [gfn* (delay (jna-base/name->global-function "tvm.codegen.llvm.target_arm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} target_x86-64
(let [gfn* (delay (jna-base/name->global-function "tvm.codegen.llvm.target_x86-64"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

