(ns tvm-clj.jna.fns.device_api
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cpu
(let [gfn* (delay (jna-base/name->global-function "device_api.cpu"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} rpc
(let [gfn* (delay (jna-base/name->global-function "device_api.rpc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

