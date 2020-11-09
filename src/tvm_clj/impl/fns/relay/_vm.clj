(ns tvm-clj.impl.fns.relay._vm
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private _VMCompiler-fnptr* (delay (base/name->global-function "relay._vm._VMCompiler")))
(defn _VMCompiler
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._vm._VMCompiler"}
   (apply base/call-function @_VMCompiler-fnptr* args)))

