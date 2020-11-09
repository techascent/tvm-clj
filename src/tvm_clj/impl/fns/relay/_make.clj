(ns tvm-clj.impl.fns.relay._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ConstructorValue-fnptr* (delay (base/name->global-function "relay._make.ConstructorValue")))
(defn ConstructorValue
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._make.ConstructorValue"}
   (apply base/call-function @ConstructorValue-fnptr* args)))

(defonce ^:private RefValue-fnptr* (delay (base/name->global-function "relay._make.RefValue")))
(defn RefValue
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._make.RefValue"}
   (apply base/call-function @RefValue-fnptr* args)))

(defonce ^:private reinterpret-fnptr* (delay (base/name->global-function "relay._make.reinterpret")))
(defn reinterpret
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._make.reinterpret"}
   (apply base/call-function @reinterpret-fnptr* args)))

