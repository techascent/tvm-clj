(ns tvm-clj.impl.fns.topi.cuda
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private dense_cuda-fnptr* (delay (base/name->global-function "topi.cuda.dense_cuda")))
(defn dense_cuda
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.dense_cuda"}
   (apply base/call-function @dense_cuda-fnptr* args)))

(defonce ^:private schedule_dense-fnptr* (delay (base/name->global-function "topi.cuda.schedule_dense")))
(defn schedule_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_dense"}
   (apply base/call-function @schedule_dense-fnptr* args)))

(defonce ^:private schedule_global_pool-fnptr* (delay (base/name->global-function "topi.cuda.schedule_global_pool")))
(defn schedule_global_pool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_global_pool"}
   (apply base/call-function @schedule_global_pool-fnptr* args)))

(defonce ^:private schedule_injective-fnptr* (delay (base/name->global-function "topi.cuda.schedule_injective")))
(defn schedule_injective
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_injective"}
   (apply base/call-function @schedule_injective-fnptr* args)))

(defonce ^:private schedule_injective_from_existing-fnptr* (delay (base/name->global-function "topi.cuda.schedule_injective_from_existing")))
(defn schedule_injective_from_existing
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_injective_from_existing"}
   (apply base/call-function @schedule_injective_from_existing-fnptr* args)))

(defonce ^:private schedule_lrn-fnptr* (delay (base/name->global-function "topi.cuda.schedule_lrn")))
(defn schedule_lrn
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_lrn"}
   (apply base/call-function @schedule_lrn-fnptr* args)))

(defonce ^:private schedule_pool-fnptr* (delay (base/name->global-function "topi.cuda.schedule_pool")))
(defn schedule_pool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_pool"}
   (apply base/call-function @schedule_pool-fnptr* args)))

(defonce ^:private schedule_reduce-fnptr* (delay (base/name->global-function "topi.cuda.schedule_reduce")))
(defn schedule_reduce
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_reduce"}
   (apply base/call-function @schedule_reduce-fnptr* args)))

(defonce ^:private schedule_softmax-fnptr* (delay (base/name->global-function "topi.cuda.schedule_softmax")))
(defn schedule_softmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cuda.schedule_softmax"}
   (apply base/call-function @schedule_softmax-fnptr* args)))

