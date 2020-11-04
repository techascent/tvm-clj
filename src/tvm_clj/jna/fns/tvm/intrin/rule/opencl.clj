(ns tvm-clj.jna.fns.tvm.intrin.rule.opencl
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ceil
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.ceil"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cos
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.cos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cosh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.cosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp10
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.exp10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp2
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.exp2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fabs
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.fabs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.floor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fmod
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.fmod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.log"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log10
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.log10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log2
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.log2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} popcount
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.popcount"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} pow
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.pow"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} round
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.round"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sin
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.sin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sinh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.sinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sqrt
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.sqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tanh
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} trunc
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.trunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tvm_warp_shuffle
(let [gfn* (delay (jna-base/name->global-function "tvm.intrin.rule.opencl.tvm_warp_shuffle"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

