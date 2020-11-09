(ns tvm-clj.impl.fns.device_api
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private cpu-fnptr* (delay (base/name->global-function "device_api.cpu")))
(defn cpu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.cpu"}
   (apply base/call-function @cpu-fnptr* args)))

(defonce ^:private rpc-fnptr* (delay (base/name->global-function "device_api.rpc")))
(defn rpc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.rpc"}
   (apply base/call-function @rpc-fnptr* args)))

