(ns tvm-clj.jna.fns.tvm.contrib.sort
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argsort
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.sort.argsort"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argsort_nms
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.sort.argsort_nms"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} topk
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.sort.topk"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

