(ns tvm-clj.jna.fns.tvm.intrin.rule.aocl_sw_emu
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.fabs"))]
  (defn fabs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.fabs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.popcount"))]
  (defn popcount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.popcount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.pow"))]
  (defn pow
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.pow"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.aocl_sw_emu.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.intrin.rule.aocl_sw_emu.trunc"}
     (apply jna-base/call-function @gfn* args))))

