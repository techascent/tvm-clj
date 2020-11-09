(ns tvm-clj.jna.fns.relay.op
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op._OpImplementationCompute"))]
  (defn _OpImplementationCompute
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._OpImplementationCompute"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._OpImplementationSchedule"))]
  (defn _OpImplementationSchedule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._OpImplementationSchedule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._OpStrategyAddImplementation"))]
  (defn _OpStrategyAddImplementation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._OpStrategyAddImplementation"}
     (apply jna-base/call-function @gfn* args))))

