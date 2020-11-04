(ns tvm-clj.jna.fns.tvm.intrin.rule.llvm
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ceil
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.ceil"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cos
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.cos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cosh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.cosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp10
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.exp10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp2
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.exp2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fabs
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.fabs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.floor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fma
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.fma"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.log"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log10
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.log10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log2
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.log2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} nearbyint
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.nearbyint"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} popcount
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.popcount"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pow
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.pow"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} prefetch
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.prefetch"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} round
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.round"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sin
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.sin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sinh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.sinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sqrt
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.sqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tan
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.tan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tanh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} trunc
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.llvm.trunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

