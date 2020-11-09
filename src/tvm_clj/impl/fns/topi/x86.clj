(ns tvm-clj.impl.fns.topi.x86
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private default_schedule-fnptr* (delay (base/name->global-function "topi.x86.default_schedule")))
(defn default_schedule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.x86.default_schedule"}
   (apply base/call-function @default_schedule-fnptr* args)))

(defonce ^:private schedule_binarize_pack-fnptr* (delay (base/name->global-function "topi.x86.schedule_binarize_pack")))
(defn schedule_binarize_pack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.x86.schedule_binarize_pack"}
   (apply base/call-function @schedule_binarize_pack-fnptr* args)))

(defonce ^:private schedule_binary_dense-fnptr* (delay (base/name->global-function "topi.x86.schedule_binary_dense")))
(defn schedule_binary_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.x86.schedule_binary_dense"}
   (apply base/call-function @schedule_binary_dense-fnptr* args)))

(defonce ^:private schedule_injective-fnptr* (delay (base/name->global-function "topi.x86.schedule_injective")))
(defn schedule_injective
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.x86.schedule_injective"}
   (apply base/call-function @schedule_injective-fnptr* args)))

(defonce ^:private schedule_injective_from_existing-fnptr* (delay (base/name->global-function "topi.x86.schedule_injective_from_existing")))
(defn schedule_injective_from_existing
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.x86.schedule_injective_from_existing"}
   (apply base/call-function @schedule_injective_from_existing-fnptr* args)))

