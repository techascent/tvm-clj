(ns tvm-clj.impl.fns.tvm.contrib.sort
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private argsort-fnptr* (delay (base/name->global-function "tvm.contrib.sort.argsort")))
(defn argsort
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.sort.argsort"}
   (apply base/call-function @argsort-fnptr* args)))

(defonce ^:private argsort_nms-fnptr* (delay (base/name->global-function "tvm.contrib.sort.argsort_nms")))
(defn argsort_nms
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.sort.argsort_nms"}
   (apply base/call-function @argsort_nms-fnptr* args)))

(defonce ^:private topk-fnptr* (delay (base/name->global-function "tvm.contrib.sort.topk")))
(defn topk
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.sort.topk"}
   (apply base/call-function @topk-fnptr* args)))

