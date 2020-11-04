(ns tvm-clj.jna.fns.tvm.intrin.rule.sdaccel
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.cos"))]
  (defn cos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.cos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.cosh"))]
  (defn cosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.cosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.exp10"))]
  (defn exp10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.exp10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.exp2"))]
  (defn exp2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.exp2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.fabs"))]
  (defn fabs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.fabs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.log10"))]
  (defn log10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.log10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.log2"))]
  (defn log2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.log2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.popcount"))]
  (defn popcount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.popcount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.pow"))]
  (defn pow
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.pow"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.sin"))]
  (defn sin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.sin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.sinh"))]
  (defn sinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.sinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.sdaccel.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.sdaccel.trunc"}
     (apply jna-base/call-function @gfn* args))))

