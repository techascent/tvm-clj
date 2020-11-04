(ns tvm-clj.jna.fns.relay.op.dyn.image._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} resize
(let [gfn* (delay (jna-base/name->global-function "relay.op.dyn.image._make.resize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

