(ns tvm-clj.jna.fns.tir.analysis
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} expr_deep_equal
(let [gfn* (delay (jna-base/name->global-function "tir.analysis.expr_deep_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} verify_gpu_code
(let [gfn* (delay (jna-base/name->global-function "tir.analysis.verify_gpu_code"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} verify_memory
(let [gfn* (delay (jna-base/name->global-function "tir.analysis.verify_memory"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} verify_ssa
(let [gfn* (delay (jna-base/name->global-function "tir.analysis.verify_ssa"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

