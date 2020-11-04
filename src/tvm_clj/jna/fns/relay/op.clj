(ns tvm-clj.jna.fns.relay.op
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpImplementationCompute
(let [gfn* (delay (jna-base/name->global-function "relay.op._OpImplementationCompute"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpImplementationSchedule
(let [gfn* (delay (jna-base/name->global-function "relay.op._OpImplementationSchedule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpStrategyAddImplementation
(let [gfn* (delay (jna-base/name->global-function "relay.op._OpStrategyAddImplementation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

