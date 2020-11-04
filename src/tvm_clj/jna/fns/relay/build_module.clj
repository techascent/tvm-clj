(ns tvm-clj.jna.fns.relay.build_module
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BindParamsByName
(let [gfn* (delay (jna-base/name->global-function "relay.build_module.BindParamsByName"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _BuildModule
(let [gfn* (delay (jna-base/name->global-function "relay.build_module._BuildModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _GraphRuntimeCodegen
(let [gfn* (delay (jna-base/name->global-function "relay.build_module._GraphRuntimeCodegen"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

