(ns tvm-clj.impl.fns.hybrid
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private _Dump-fnptr* (delay (base/name->global-function "hybrid._Dump")))
(defn _Dump
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "hybrid._Dump"}
   (apply base/call-function @_Dump-fnptr* args)))

