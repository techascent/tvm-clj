(ns tvm-clj.jna.fns.relay.build_module
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.build_module.BindParamsByName"))]
  (defn BindParamsByName
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.build_module.BindParamsByName"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.build_module._BuildModule"))]
  (defn _BuildModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.build_module._BuildModule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.build_module._GraphRuntimeCodegen"))]
  (defn _GraphRuntimeCodegen
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.build_module._GraphRuntimeCodegen"}
     (apply jna-base/call-function @gfn* args))))

