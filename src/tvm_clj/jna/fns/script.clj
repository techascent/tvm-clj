(ns tvm-clj.jna.fns.script
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AsTVMScript
(let [gfn* (delay (jna-base/name->global-function "script.AsTVMScript"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

