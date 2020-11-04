(ns tvm-clj.jna.fns.relay._vm
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _VMCompiler
(let [gfn* (delay (jna-base/name->global-function "relay._vm._VMCompiler"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

