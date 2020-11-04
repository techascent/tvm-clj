(ns tvm-clj.jna.fns.relay.op.dyn.image._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.image._make.resize"))]
  (defn resize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.dyn.image._make.resize"}
     (apply jna-base/call-function @gfn* args))))

