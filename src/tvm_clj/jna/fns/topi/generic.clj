(ns tvm-clj.jna.fns.topi.generic
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} default_schedule
(let [gfn* (delay (jna-base/name->global-function "topi.generic.default_schedule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_extern
(let [gfn* (delay (jna-base/name->global-function "topi.generic.schedule_extern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_injective
(let [gfn* (delay (jna-base/name->global-function "topi.generic.schedule_injective"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} schedule_injective_from_existing
(let [gfn* (delay (jna-base/name->global-function "topi.generic.schedule_injective_from_existing"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

