(ns tvm-clj.impl.fns.tvm.graph_runtime
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private create-fnptr* (delay (base/name->global-function "tvm.graph_runtime.create")))
(defn create
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.graph_runtime.create"}
   (apply base/call-function @create-fnptr* args)))

