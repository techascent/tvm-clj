(ns tvm-clj.impl.fns.tvm.intrin.rule.hexagon
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private fma-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.fma")))
(defn fma
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.fma"}
   (apply base/call-function @fma-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private popcount-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.popcount")))
(defn popcount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.popcount"}
   (apply base/call-function @popcount-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.hexagon.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.hexagon.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

