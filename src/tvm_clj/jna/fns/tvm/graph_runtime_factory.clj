(ns tvm-clj.jna.fns.tvm.graph_runtime_factory
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.graph_runtime_factory.create"))]
  (defn create
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.graph_runtime_factory.create"}
     (apply jna-base/call-function @gfn* args))))

