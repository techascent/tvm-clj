(ns tvm-clj.impl.fns.script
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AsTVMScript-fnptr* (delay (base/name->global-function "script.AsTVMScript")))
(defn AsTVMScript
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "script.AsTVMScript"}
   (apply base/call-function @AsTVMScript-fnptr* args)))

