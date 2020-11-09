(ns tvm-clj.impl.fns.tvm.intrin.rule.nvptx
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private atan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.atan")))
(defn atan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.atan"}
   (apply base/call-function @atan-fnptr* args)))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private erf-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.erf")))
(defn erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.erf"}
   (apply base/call-function @erf-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private exp10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.exp10")))
(defn exp10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.exp10"}
   (apply base/call-function @exp10-fnptr* args)))

(defonce ^:private exp2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.exp2")))
(defn exp2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.exp2"}
   (apply base/call-function @exp2-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private fma-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.fma")))
(defn fma
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.fma"}
   (apply base/call-function @fma-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.nvptx.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.nvptx.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

