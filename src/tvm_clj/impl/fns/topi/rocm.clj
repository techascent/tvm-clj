(ns tvm-clj.jna.fns.topi.rocm
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.dense_cuda"))]
  (defn dense_cuda
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.dense_cuda"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_dense"))]
  (defn schedule_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_global_pool"))]
  (defn schedule_global_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_global_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_injective"))]
  (defn schedule_injective
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_injective"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_injective_from_existing"))]
  (defn schedule_injective_from_existing
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_injective_from_existing"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_lrn"))]
  (defn schedule_lrn
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_lrn"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_pool"))]
  (defn schedule_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_reduce"))]
  (defn schedule_reduce
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_reduce"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rocm.schedule_softmax"))]
  (defn schedule_softmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rocm.schedule_softmax"}
     (apply jna-base/call-function @gfn* args))))

