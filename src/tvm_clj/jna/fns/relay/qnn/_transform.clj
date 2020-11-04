(ns tvm-clj.jna.fns.relay.qnn._transform
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Legalize
(let [gfn* (delay (jna-base/name->global-function "relay.qnn._transform.Legalize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

