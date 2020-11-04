(ns tvm-clj.jna.fns.tvm.contrib.random
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} normal
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.normal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} randint
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.randint"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} random_fill
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.random_fill"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} uniform
(let [gfn* (delay (jna-base/name->global-function "tvm.contrib.random.uniform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

