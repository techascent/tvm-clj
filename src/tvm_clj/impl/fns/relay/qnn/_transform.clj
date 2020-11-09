(ns tvm-clj.impl.fns.relay.qnn._transform
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private Legalize-fnptr* (delay (base/name->global-function "relay.qnn._transform.Legalize")))
(defn Legalize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn._transform.Legalize"}
   (apply base/call-function @Legalize-fnptr* args)))

