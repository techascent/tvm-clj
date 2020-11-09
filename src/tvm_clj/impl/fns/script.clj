(ns tvm-clj.jna.fns.script
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "script.AsTVMScript"))]
  (defn AsTVMScript
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "script.AsTVMScript"}
     (apply jna-base/call-function @gfn* args))))

