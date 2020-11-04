(ns tvm-clj.jna.fns.test.op
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InferTensorizeRegion
(let [gfn* (delay (jna-base/name->global-function "test.op.InferTensorizeRegion"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MatchTensorizeBody
(let [gfn* (delay (jna-base/name->global-function "test.op.MatchTensorizeBody"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

