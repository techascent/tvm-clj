(ns tvm-clj.impl.fns.tvm.contrib.mkl
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private batch_matmul-fnptr* (delay (base/name->global-function "tvm.contrib.mkl.batch_matmul")))
(defn batch_matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.mkl.batch_matmul"}
   (apply base/call-function @batch_matmul-fnptr* args)))

(defonce ^:private batch_matmul_iterative-fnptr* (delay (base/name->global-function "tvm.contrib.mkl.batch_matmul_iterative")))
(defn batch_matmul_iterative
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.mkl.batch_matmul_iterative"}
   (apply base/call-function @batch_matmul_iterative-fnptr* args)))

(defonce ^:private matmul-fnptr* (delay (base/name->global-function "tvm.contrib.mkl.matmul")))
(defn matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.mkl.matmul"}
   (apply base/call-function @matmul-fnptr* args)))

(defonce ^:private matmul_u8s8s32-fnptr* (delay (base/name->global-function "tvm.contrib.mkl.matmul_u8s8s32")))
(defn matmul_u8s8s32
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.mkl.matmul_u8s8s32"}
   (apply base/call-function @matmul_u8s8s32-fnptr* args)))

