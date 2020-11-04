(ns tvm-clj.jna.fns.support
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetLibInfo
(let [gfn* (delay (jna-base/name->global-function "support.GetLibInfo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

