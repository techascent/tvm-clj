(ns tvm-clj.jna.fns.tvm.intrin.rule.aocl_sw_emu
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ceil
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.ceil"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fabs
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.fabs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.floor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.log"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} popcount
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.popcount"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pow
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.pow"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} round
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.round"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sqrt
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.sqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tanh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} trunc
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.trunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

