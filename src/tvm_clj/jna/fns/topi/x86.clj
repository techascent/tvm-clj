(ns tvm-clj.jna.fns.topi.x86
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.x86.default_schedule"))]
  (defn default_schedule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.x86.default_schedule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_binarize_pack"))]
  (defn schedule_binarize_pack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.x86.schedule_binarize_pack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_binary_dense"))]
  (defn schedule_binary_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.x86.schedule_binary_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_injective"))]
  (defn schedule_injective
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.x86.schedule_injective"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_injective_from_existing"))]
  (defn schedule_injective_from_existing
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.x86.schedule_injective_from_existing"}
     (apply jna-base/call-function @gfn* args))))

