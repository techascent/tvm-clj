(ns tvm-clj.impl.fns.relay.op
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private _OpImplementationCompute-fnptr* (delay (base/name->global-function "relay.op._OpImplementationCompute")))
(defn _OpImplementationCompute
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._OpImplementationCompute"}
   (apply base/call-function @_OpImplementationCompute-fnptr* args)))

(defonce ^:private _OpImplementationSchedule-fnptr* (delay (base/name->global-function "relay.op._OpImplementationSchedule")))
(defn _OpImplementationSchedule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._OpImplementationSchedule"}
   (apply base/call-function @_OpImplementationSchedule-fnptr* args)))

(defonce ^:private _OpStrategyAddImplementation-fnptr* (delay (base/name->global-function "relay.op._OpStrategyAddImplementation")))
(defn _OpStrategyAddImplementation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._OpStrategyAddImplementation"}
   (apply base/call-function @_OpStrategyAddImplementation-fnptr* args)))

