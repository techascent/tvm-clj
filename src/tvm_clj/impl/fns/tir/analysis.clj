(ns tvm-clj.impl.fns.tir.analysis
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private expr_deep_equal-fnptr* (delay (base/name->global-function "tir.analysis.expr_deep_equal")))
(defn expr_deep_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.analysis.expr_deep_equal"}
   (apply base/call-function @expr_deep_equal-fnptr* args)))

(defonce ^:private verify_gpu_code-fnptr* (delay (base/name->global-function "tir.analysis.verify_gpu_code")))
(defn verify_gpu_code
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.analysis.verify_gpu_code"}
   (apply base/call-function @verify_gpu_code-fnptr* args)))

(defonce ^:private verify_memory-fnptr* (delay (base/name->global-function "tir.analysis.verify_memory")))
(defn verify_memory
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.analysis.verify_memory"}
   (apply base/call-function @verify_memory-fnptr* args)))

(defonce ^:private verify_ssa-fnptr* (delay (base/name->global-function "tir.analysis.verify_ssa")))
(defn verify_ssa
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.analysis.verify_ssa"}
   (apply base/call-function @verify_ssa-fnptr* args)))

