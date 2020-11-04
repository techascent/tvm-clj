(ns tvm-clj.jna.fns.relay.ext
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.ext.ccompiler"))]
  (defn ccompiler
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ext.ccompiler"}
     (apply jna-base/call-function @gfn* args))))

