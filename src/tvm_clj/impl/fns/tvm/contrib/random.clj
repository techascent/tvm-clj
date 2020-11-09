(ns tvm-clj.impl.fns.tvm.contrib.random
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private normal-fnptr* (delay (base/name->global-function "tvm.contrib.random.normal")))
(defn normal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.random.normal"}
   (apply base/call-function @normal-fnptr* args)))

(defonce ^:private randint-fnptr* (delay (base/name->global-function "tvm.contrib.random.randint")))
(defn randint
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.random.randint"}
   (apply base/call-function @randint-fnptr* args)))

(defonce ^:private random_fill-fnptr* (delay (base/name->global-function "tvm.contrib.random.random_fill")))
(defn random_fill
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.random.random_fill"}
   (apply base/call-function @random_fill-fnptr* args)))

(defonce ^:private uniform-fnptr* (delay (base/name->global-function "tvm.contrib.random.uniform")))
(defn uniform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.contrib.random.uniform"}
   (apply base/call-function @uniform-fnptr* args)))

