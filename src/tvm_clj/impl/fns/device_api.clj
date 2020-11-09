(ns tvm-clj.jna.fns.device_api
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "device_api.cpu"))]
  (defn cpu
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "device_api.cpu"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "device_api.rpc"))]
  (defn rpc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "device_api.rpc"}
     (apply jna-base/call-function @gfn* args))))

