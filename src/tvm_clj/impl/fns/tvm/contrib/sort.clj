(ns tvm-clj.jna.fns.tvm.contrib.sort
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.sort.argsort"))]
  (defn argsort
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.sort.argsort"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.sort.argsort_nms"))]
  (defn argsort_nms
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.sort.argsort_nms"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.sort.topk"))]
  (defn topk
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.sort.topk"}
     (apply jna-base/call-function @gfn* args))))

