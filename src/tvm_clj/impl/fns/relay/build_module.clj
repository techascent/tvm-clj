(ns tvm-clj.impl.fns.relay.build_module
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private BindParamsByName-fnptr* (delay (base/name->global-function "relay.build_module.BindParamsByName")))
(defn BindParamsByName
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.build_module.BindParamsByName"}
   (apply base/call-function @BindParamsByName-fnptr* args)))

(defonce ^:private _BuildModule-fnptr* (delay (base/name->global-function "relay.build_module._BuildModule")))
(defn _BuildModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.build_module._BuildModule"}
   (apply base/call-function @_BuildModule-fnptr* args)))

(defonce ^:private _GraphRuntimeCodegen-fnptr* (delay (base/name->global-function "relay.build_module._GraphRuntimeCodegen")))
(defn _GraphRuntimeCodegen
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.build_module._GraphRuntimeCodegen"}
   (apply base/call-function @_GraphRuntimeCodegen-fnptr* args)))

