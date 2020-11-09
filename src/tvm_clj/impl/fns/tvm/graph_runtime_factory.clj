(ns tvm-clj.impl.fns.tvm.graph_runtime_factory
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private create-fnptr* (delay (base/name->global-function "tvm.graph_runtime_factory.create")))
(defn create
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.graph_runtime_factory.create"}
   (apply base/call-function @create-fnptr* args)))

