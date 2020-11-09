(ns tvm-clj.impl.fns.tvm.intrin.rule.llvm
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private exp10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.exp10")))
(defn exp10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.exp10"}
   (apply base/call-function @exp10-fnptr* args)))

(defonce ^:private exp2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.exp2")))
(defn exp2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.exp2"}
   (apply base/call-function @exp2-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private fma-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.fma")))
(defn fma
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.fma"}
   (apply base/call-function @fma-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private nearbyint-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.nearbyint")))
(defn nearbyint
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.nearbyint"}
   (apply base/call-function @nearbyint-fnptr* args)))

(defonce ^:private popcount-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.popcount")))
(defn popcount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.popcount"}
   (apply base/call-function @popcount-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private prefetch-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.prefetch")))
(defn prefetch
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.prefetch"}
   (apply base/call-function @prefetch-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.llvm.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.llvm.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

