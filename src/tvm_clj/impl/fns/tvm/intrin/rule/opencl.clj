(ns tvm-clj.impl.fns.tvm.intrin.rule.opencl
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private exp10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.exp10")))
(defn exp10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.exp10"}
   (apply base/call-function @exp10-fnptr* args)))

(defonce ^:private exp2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.exp2")))
(defn exp2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.exp2"}
   (apply base/call-function @exp2-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private fmod-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.fmod")))
(defn fmod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.fmod"}
   (apply base/call-function @fmod-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private popcount-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.popcount")))
(defn popcount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.popcount"}
   (apply base/call-function @popcount-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

(defonce ^:private tvm_warp_shuffle-fnptr* (delay (base/name->global-function "tvm.intrin.rule.opencl.tvm_warp_shuffle")))
(defn tvm_warp_shuffle
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.opencl.tvm_warp_shuffle"}
   (apply base/call-function @tvm_warp_shuffle-fnptr* args)))

