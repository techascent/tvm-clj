(ns tvm-clj.jna.fns.tvm.intrin.rule.llvm
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.cos"))]
  (defn cos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.cos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.cosh"))]
  (defn cosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.cosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.exp10"))]
  (defn exp10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.exp10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.exp2"))]
  (defn exp2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.exp2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.fabs"))]
  (defn fabs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.fabs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.fma"))]
  (defn fma
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.fma"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.log10"))]
  (defn log10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.log10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.log2"))]
  (defn log2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.log2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.nearbyint"))]
  (defn nearbyint
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.nearbyint"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.popcount"))]
  (defn popcount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.popcount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.pow"))]
  (defn pow
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.pow"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.prefetch"))]
  (defn prefetch
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.prefetch"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.sin"))]
  (defn sin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.sin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.sinh"))]
  (defn sinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.sinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.tan"))]
  (defn tan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.tan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.llvm.trunc"}
     (apply jna-base/call-function @gfn* args))))

