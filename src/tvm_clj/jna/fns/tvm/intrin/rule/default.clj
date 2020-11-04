(ns tvm-clj.jna.fns.tvm.intrin.rule.default
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} acos
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.acos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} acosh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.acosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} asin
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.asin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} asinh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.asinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atan
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.atan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atan2
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.atan2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atanh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.atanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} copysign
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.copysign"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cos
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.cos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cosh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.cosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} erf
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.erf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} hypot
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.hypot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isfinite
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.isfinite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isinf
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.isinf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ldexp
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.ldexp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.log"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log10
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.log10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log1p
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.log1p"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log2
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.log2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} nextafter
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.nextafter"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pow
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.pow"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} q_multiply_shift
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.q_multiply_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} rsqrt
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.rsqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sigmoid
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.sigmoid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sin
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.sin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sinh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.sinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sqrt
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.sqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tan
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.tan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tanh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.default.tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

