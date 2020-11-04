(ns tvm-clj.jna.fns.topi.vision
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reorg
(let [gfn* (delay (jna-base/name->global-function "topi.vision.reorg"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

