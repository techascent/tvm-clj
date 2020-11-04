(ns tvm-clj.jna.fns.tvm.graph_runtime
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.graph_runtime.create"))]
  (defn create
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.graph_runtime.create"}
     (apply jna-base/call-function @gfn* args))))

