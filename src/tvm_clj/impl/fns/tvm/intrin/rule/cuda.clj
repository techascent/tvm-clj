(ns tvm-clj.impl.fns.tvm.intrin.rule.cuda
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private atan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.atan")))
(defn atan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.atan"}
   (apply base/call-function @atan-fnptr* args)))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private erf-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.erf")))
(defn erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.erf"}
   (apply base/call-function @erf-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private exp10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.exp10")))
(defn exp10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.exp10"}
   (apply base/call-function @exp10-fnptr* args)))

(defonce ^:private exp2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.exp2")))
(defn exp2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.exp2"}
   (apply base/call-function @exp2-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private fmod-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.fmod")))
(defn fmod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.fmod"}
   (apply base/call-function @fmod-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private popcount-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.popcount")))
(defn popcount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.popcount"}
   (apply base/call-function @popcount-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

(defonce ^:private tvm_warp_activemask-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_activemask")))
(defn tvm_warp_activemask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.tvm_warp_activemask"}
   (apply base/call-function @tvm_warp_activemask-fnptr* args)))

(defonce ^:private tvm_warp_shuffle-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_shuffle")))
(defn tvm_warp_shuffle
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.tvm_warp_shuffle"}
   (apply base/call-function @tvm_warp_shuffle-fnptr* args)))

(defonce ^:private tvm_warp_shuffle_down-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_shuffle_down")))
(defn tvm_warp_shuffle_down
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.tvm_warp_shuffle_down"}
   (apply base/call-function @tvm_warp_shuffle_down-fnptr* args)))

(defonce ^:private tvm_warp_shuffle_up-fnptr* (delay (base/name->global-function "tvm.intrin.rule.cuda.tvm_warp_shuffle_up")))
(defn tvm_warp_shuffle_up
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.cuda.tvm_warp_shuffle_up"}
   (apply base/call-function @tvm_warp_shuffle_up-fnptr* args)))

