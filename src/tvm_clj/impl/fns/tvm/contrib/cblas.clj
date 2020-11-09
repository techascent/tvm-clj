(ns tvm-clj.impl.fns.tvm.contrib.cblas
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private batch_matmul-fnptr* (delay (base/name->global-function "tvm.contrib.cblas.batch_matmul")))
(defn batch_matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.cblas.batch_matmul"}
   (apply base/call-function @batch_matmul-fnptr* args)))

(defonce ^:private batch_matmul_iterative-fnptr* (delay (base/name->global-function "tvm.contrib.cblas.batch_matmul_iterative")))
(defn batch_matmul_iterative
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.cblas.batch_matmul_iterative"}
   (apply base/call-function @batch_matmul_iterative-fnptr* args)))

(defonce ^:private matmul-fnptr* (delay (base/name->global-function "tvm.contrib.cblas.matmul")))
(defn matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.cblas.matmul"}
   (apply base/call-function @matmul-fnptr* args)))

