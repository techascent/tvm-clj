(ns tvm-clj.impl.fns.tvm.intrin.rule.rocm
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private atan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.atan")))
(defn atan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.atan"}
   (apply base/call-function @atan-fnptr* args)))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private erf-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.erf")))
(defn erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.erf"}
   (apply base/call-function @erf-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private exp10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.exp10")))
(defn exp10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.exp10"}
   (apply base/call-function @exp10-fnptr* args)))

(defonce ^:private exp2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.exp2")))
(defn exp2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.exp2"}
   (apply base/call-function @exp2-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private fma-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.fma")))
(defn fma
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.fma"}
   (apply base/call-function @fma-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

(defonce ^:private tvm_warp_activemask-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.tvm_warp_activemask")))
(defn tvm_warp_activemask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.tvm_warp_activemask"}
   (apply base/call-function @tvm_warp_activemask-fnptr* args)))

(defonce ^:private tvm_warp_shuffle-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.tvm_warp_shuffle")))
(defn tvm_warp_shuffle
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.tvm_warp_shuffle"}
   (apply base/call-function @tvm_warp_shuffle-fnptr* args)))

(defonce ^:private tvm_warp_shuffle_down-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.tvm_warp_shuffle_down")))
(defn tvm_warp_shuffle_down
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.tvm_warp_shuffle_down"}
   (apply base/call-function @tvm_warp_shuffle_down-fnptr* args)))

(defonce ^:private tvm_warp_shuffle_up-fnptr* (delay (base/name->global-function "tvm.intrin.rule.rocm.tvm_warp_shuffle_up")))
(defn tvm_warp_shuffle_up
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.rocm.tvm_warp_shuffle_up"}
   (apply base/call-function @tvm_warp_shuffle_up-fnptr* args)))

