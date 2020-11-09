(ns tvm-clj.impl.fns.tvm.intrin.rule.aocl
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private fabs-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.fabs")))
(defn fabs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.fabs"}
   (apply base/call-function @fabs-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private popcount-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.popcount")))
(defn popcount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.popcount"}
   (apply base/call-function @popcount-fnptr* args)))

(defonce ^:private pow-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.pow")))
(defn pow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.pow"}
   (apply base/call-function @pow-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tvm.intrin.rule.aocl.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.intrin.rule.aocl.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

