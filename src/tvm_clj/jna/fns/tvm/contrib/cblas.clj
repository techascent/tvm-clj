(ns tvm-clj.jna.fns.tvm.contrib.cblas
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} batch_matmul
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.cblas.batch_matmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} batch_matmul_iterative
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.cblas.batch_matmul_iterative"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} matmul
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.cblas.matmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

