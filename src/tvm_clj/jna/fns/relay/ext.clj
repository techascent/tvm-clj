(ns tvm-clj.jna.fns.relay.ext
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ccompiler
(let [gfn* (delay (jna-base/name->global-function "relay.ext.ccompiler"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

