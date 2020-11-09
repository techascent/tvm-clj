(ns tvm-clj.impl.fns.tvm.intrin.rule.default
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private acos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.acos")))
(defn acos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.acos"}
   (apply base/call-function @acos-fnptr* args)))

(defonce ^:private acosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.acosh")))
(defn acosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.acosh"}
   (apply base/call-function @acosh-fnptr* args)))

(defonce ^:private asin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.asin")))
(defn asin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.asin"}
   (apply base/call-function @asin-fnptr* args)))

(defonce ^:private asinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.asinh")))
(defn asinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.asinh"}
   (apply base/call-function @asinh-fnptr* args)))

(defonce ^:private atan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.atan")))
(defn atan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.atan"}
   (apply base/call-function @atan-fnptr* args)))

(defonce ^:private atan2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.atan2")))
(defn atan2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.atan2"}
   (apply base/call-function @atan2-fnptr* args)))

(defonce ^:private atanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.atanh")))
(defn atanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.atanh"}
   (apply base/call-function @atanh-fnptr* args)))

(defonce ^:private copysign-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.copysign")))
(defn copysign
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.copysign"}
   (apply base/call-function @copysign-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private erf-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.erf")))
(defn erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.erf"}
   (apply base/call-function @erf-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private hypot-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.hypot")))
(defn hypot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.hypot"}
   (apply base/call-function @hypot-fnptr* args)))

(defonce ^:private isfinite-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.isfinite")))
(defn isfinite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.isfinite"}
   (apply base/call-function @isfinite-fnptr* args)))

(defonce ^:private isinf-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.isinf")))
(defn isinf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.isinf"}
   (apply base/call-function @isinf-fnptr* args)))

(defonce ^:private ldexp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.ldexp")))
(defn ldexp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.ldexp"}
   (apply base/call-function @ldexp-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log1p-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.log1p")))
(defn log1p
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.log1p"}
   (apply base/call-function @log1p-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private nextafter-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.nextafter")))
(defn nextafter
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.nextafter"}
   (apply base/call-function @nextafter-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private q_multiply_shift-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.q_multiply_shift")))
(defn q_multiply_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.q_multiply_shift"}
   (apply base/call-function @q_multiply_shift-fnptr* args)))

(defonce ^:private rsqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.rsqrt")))
(defn rsqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.rsqrt"}
   (apply base/call-function @rsqrt-fnptr* args)))

(defonce ^:private sigmoid-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.sigmoid")))
(defn sigmoid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.sigmoid"}
   (apply base/call-function @sigmoid-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.default.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.default.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

