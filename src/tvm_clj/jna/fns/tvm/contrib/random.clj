(ns tvm-clj.jna.fns.tvm.contrib.random
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.normal"))]
  (defn normal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.random.normal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.randint"))]
  (defn randint
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.random.randint"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.random_fill"))]
  (defn random_fill
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.random.random_fill"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.uniform"))]
  (defn uniform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.contrib.random.uniform"}
     (apply jna-base/call-function @gfn* args))))

