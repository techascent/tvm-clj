(ns tvm-clj.impl.fns.relay.op.dyn.image._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private resize-fnptr* (delay (base/name->global-function "relay.op.dyn.image._make.resize")))
(defn resize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.dyn.image._make.resize"}
   (apply base/call-function @resize-fnptr* args)))

