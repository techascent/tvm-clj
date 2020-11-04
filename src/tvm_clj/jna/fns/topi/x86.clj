(ns tvm-clj.jna.fns.topi.x86
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} default_schedule
(let [gfn* (delay (jna-base/name->global-function "topi.x86.default_schedule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_binarize_pack
(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_binarize_pack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_binary_dense
(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_binary_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_injective
(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_injective"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_injective_from_existing
(let [gfn* (delay (jna-base/name->global-function "topi.x86.schedule_injective_from_existing"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

