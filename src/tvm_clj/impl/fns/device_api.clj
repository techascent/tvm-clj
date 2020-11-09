(ns tvm-clj.impl.fns.device_api
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private cpu-fnptr* (delay (base/name->global-function "device_api.cpu")))
(defn cpu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.cpu"}
   (apply base/call-function @cpu-fnptr* args)))

(defonce ^:private cpu_pinned-fnptr* (delay (base/name->global-function "device_api.cpu_pinned")))
(defn cpu_pinned
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.cpu_pinned"}
   (apply base/call-function @cpu_pinned-fnptr* args)))

(defonce ^:private gpu-fnptr* (delay (base/name->global-function "device_api.gpu")))
(defn gpu
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.gpu"}
   (apply base/call-function @gpu-fnptr* args)))

(defonce ^:private opencl-fnptr* (delay (base/name->global-function "device_api.opencl")))
(defn opencl
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.opencl"}
   (apply base/call-function @opencl-fnptr* args)))

(defonce ^:private rpc-fnptr* (delay (base/name->global-function "device_api.rpc")))
(defn rpc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "device_api.rpc"}
   (apply base/call-function @rpc-fnptr* args)))

