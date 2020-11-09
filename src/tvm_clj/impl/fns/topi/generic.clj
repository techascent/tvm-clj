(ns tvm-clj.jna.fns.topi.generic
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.generic.default_schedule"))]
  (defn default_schedule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.generic.default_schedule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.generic.schedule_extern"))]
  (defn schedule_extern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.generic.schedule_extern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.generic.schedule_injective"))]
  (defn schedule_injective
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.generic.schedule_injective"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.generic.schedule_injective_from_existing"))]
  (defn schedule_injective_from_existing
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.generic.schedule_injective_from_existing"}
     (apply jna-base/call-function @gfn* args))))

