(ns tvm-clj.jna.fns.hybrid
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "hybrid._Dump"))]
  (defn _Dump
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "hybrid._Dump"}
     (apply jna-base/call-function @gfn* args))))

