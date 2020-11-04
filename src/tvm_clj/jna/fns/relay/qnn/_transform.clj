(ns tvm-clj.jna.fns.relay.qnn._transform
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn._transform.Legalize"))]
  (defn Legalize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn._transform.Legalize"}
     (apply jna-base/call-function @gfn* args))))

