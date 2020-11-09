(ns tvm-clj.jna.fns.relay._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay._make.ConstructorValue"))]
  (defn ConstructorValue
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._make.ConstructorValue"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._make.RefValue"))]
  (defn RefValue
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._make.RefValue"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._make.reinterpret"))]
  (defn reinterpret
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._make.reinterpret"}
     (apply jna-base/call-function @gfn* args))))

