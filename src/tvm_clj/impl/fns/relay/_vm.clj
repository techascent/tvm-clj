(ns tvm-clj.jna.fns.relay._vm
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay._vm._VMCompiler"))]
  (defn _VMCompiler
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._vm._VMCompiler"}
     (apply jna-base/call-function @gfn* args))))

