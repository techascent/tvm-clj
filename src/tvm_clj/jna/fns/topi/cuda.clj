(ns tvm-clj.jna.fns.topi.cuda
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dense_cuda
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.dense_cuda"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_dense
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_global_pool
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_global_pool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_injective
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_injective"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_injective_from_existing
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_injective_from_existing"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_lrn
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_lrn"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_pool
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_pool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_reduce
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_reduce"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_softmax
(let [gfn* (delay (jna-base/name->global-function "topi.cuda.schedule_softmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

