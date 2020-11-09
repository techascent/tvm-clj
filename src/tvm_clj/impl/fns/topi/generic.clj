(ns tvm-clj.impl.fns.topi.generic
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private default_schedule-fnptr* (delay (base/name->global-function "topi.generic.default_schedule")))
(defn default_schedule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.generic.default_schedule"}
   (apply base/call-function @default_schedule-fnptr* args)))

(defonce ^:private schedule_extern-fnptr* (delay (base/name->global-function "topi.generic.schedule_extern")))
(defn schedule_extern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.generic.schedule_extern"}
   (apply base/call-function @schedule_extern-fnptr* args)))

(defonce ^:private schedule_injective-fnptr* (delay (base/name->global-function "topi.generic.schedule_injective")))
(defn schedule_injective
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.generic.schedule_injective"}
   (apply base/call-function @schedule_injective-fnptr* args)))

(defonce ^:private schedule_injective_from_existing-fnptr* (delay (base/name->global-function "topi.generic.schedule_injective_from_existing")))
(defn schedule_injective_from_existing
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.generic.schedule_injective_from_existing"}
   (apply base/call-function @schedule_injective_from_existing-fnptr* args)))

