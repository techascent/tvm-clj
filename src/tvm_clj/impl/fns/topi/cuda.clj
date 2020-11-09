(ns tvm-clj.jna.fns.topi.cuda
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.dense_cuda"))]
  (defn dense_cuda
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.dense_cuda"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_dense"))]
  (defn schedule_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_global_pool"))]
  (defn schedule_global_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_global_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_injective"))]
  (defn schedule_injective
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_injective"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_injective_from_existing"))]
  (defn schedule_injective_from_existing
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_injective_from_existing"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_lrn"))]
  (defn schedule_lrn
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_lrn"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_pool"))]
  (defn schedule_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_reduce"))]
  (defn schedule_reduce
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_reduce"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_softmax"))]
  (defn schedule_softmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cuda.schedule_softmax"}
     (apply jna-base/call-function @gfn* args))))

