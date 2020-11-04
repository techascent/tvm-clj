(ns tvm-clj.jna.fns.tir.analysis
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tir.analysis.expr_deep_equal"))]
  (defn expr_deep_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.analysis.expr_deep_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.analysis.verify_gpu_code"))]
  (defn verify_gpu_code
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.analysis.verify_gpu_code"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.analysis.verify_memory"))]
  (defn verify_memory
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.analysis.verify_memory"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.analysis.verify_ssa"))]
  (defn verify_ssa
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.analysis.verify_ssa"}
     (apply jna-base/call-function @gfn* args))))

