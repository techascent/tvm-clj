(ns tvm-clj.impl.fns.relay.ext
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ccompiler-fnptr* (delay (base/name->global-function "relay.ext.ccompiler")))
(defn ccompiler
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ext.ccompiler"}
   (apply base/call-function @ccompiler-fnptr* args)))

