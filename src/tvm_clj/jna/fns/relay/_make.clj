(ns tvm-clj.jna.fns.relay._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ConstructorValue
(let [gfn* (delay (jna-base/name->global-function "relay._make.ConstructorValue"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RefValue
(let [gfn* (delay (jna-base/name->global-function "relay._make.RefValue"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reinterpret
(let [gfn* (delay (jna-base/name->global-function "relay._make.reinterpret"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

