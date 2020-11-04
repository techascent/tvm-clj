(ns tvm-clj.jna.fns.tvm.graph_runtime_factory
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} create
(let [gfn* (delay (jna-base/name->global-function "tvm.graph_runtime_factory.create"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

