(ns tvm-clj.jna.fns.tvm.intrin.rule.aocl
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.fabs"))]
  (defn fabs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.fabs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.popcount"))]
  (defn popcount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.popcount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.pow"))]
  (defn pow
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.pow"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl.trunc"}
     (apply jna-base/call-function @gfn* args))))

