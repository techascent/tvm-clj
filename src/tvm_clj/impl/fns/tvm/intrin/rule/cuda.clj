(ns tvm-clj.jna.fns.tvm.intrin.rule.cuda
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.atan"))]
  (defn atan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.atan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.cos"))]
  (defn cos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.cos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.cosh"))]
  (defn cosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.cosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.erf"))]
  (defn erf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.erf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.exp10"))]
  (defn exp10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.exp10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.exp2"))]
  (defn exp2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.exp2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.fabs"))]
  (defn fabs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.fabs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.fmod"))]
  (defn fmod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.fmod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.log10"))]
  (defn log10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.log10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.log2"))]
  (defn log2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.log2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.popcount"))]
  (defn popcount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.popcount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.pow"))]
  (defn pow
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.pow"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.sin"))]
  (defn sin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.sin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.sinh"))]
  (defn sinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.sinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.tan"))]
  (defn tan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.tan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.trunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_activemask"))]
  (defn tvm_warp_activemask
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.tvm_warp_activemask"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_shuffle"))]
  (defn tvm_warp_shuffle
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.tvm_warp_shuffle"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_shuffle_down"))]
  (defn tvm_warp_shuffle_down
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.tvm_warp_shuffle_down"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_shuffle_up"))]
  (defn tvm_warp_shuffle_up
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.cuda.tvm_warp_shuffle_up"}
     (apply jna-base/call-function @gfn* args))))

