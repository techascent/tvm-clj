(ns tvm-clj.jna.fns.hybrid
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _Dump
(let [gfn* (delay (jna-base/name->global-function "hybrid._Dump"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

