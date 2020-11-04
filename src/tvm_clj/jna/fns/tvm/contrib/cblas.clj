(ns tvm-clj.jna.fns.tvm.contrib.cblas
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.cblas.batch_matmul"))]
  (defn batch_matmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.cblas.batch_matmul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.cblas.batch_matmul_iterative"))]
  (defn batch_matmul_iterative
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.cblas.batch_matmul_iterative"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.cblas.matmul"))]
  (defn matmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.cblas.matmul"}
     (apply jna-base/call-function @gfn* args))))

