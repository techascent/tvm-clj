(ns tvm-clj.jna.fns.arith
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ConstIntBound
(let [gfn* (delay (jna-base/name->global-function "arith.ConstIntBound"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateAnalyzer
(let [gfn* (delay (jna-base/name->global-function "arith.CreateAnalyzer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DeduceBound
(let [gfn* (delay (jna-base/name->global-function "arith.DeduceBound"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DetectClipBound
(let [gfn* (delay (jna-base/name->global-function "arith.DetectClipBound"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DetectLinearEquation
(let [gfn* (delay (jna-base/name->global-function "arith.DetectLinearEquation"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DomainTouched
(let [gfn* (delay (jna-base/name->global-function "arith.DomainTouched"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntConstraints
(let [gfn* (delay (jna-base/name->global-function "arith.IntConstraints"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntConstraintsTransform
(let [gfn* (delay (jna-base/name->global-function "arith.IntConstraintsTransform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntGroupBounds
(let [gfn* (delay (jna-base/name->global-function "arith.IntGroupBounds"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntGroupBounds_FindBestRange
(let [gfn* (delay (jna-base/name->global-function "arith.IntGroupBounds_FindBestRange"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntGroupBounds_from_range
(let [gfn* (delay (jna-base/name->global-function "arith.IntGroupBounds_from_range"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntSetIsEverything
(let [gfn* (delay (jna-base/name->global-function "arith.IntSetIsEverything"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntSetIsNothing
(let [gfn* (delay (jna-base/name->global-function "arith.IntSetIsNothing"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntervalSet
(let [gfn* (delay (jna-base/name->global-function "arith.IntervalSet"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntervalSetGetMax
(let [gfn* (delay (jna-base/name->global-function "arith.IntervalSetGetMax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IntervalSetGetMin
(let [gfn* (delay (jna-base/name->global-function "arith.IntervalSetGetMin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModularSet
(let [gfn* (delay (jna-base/name->global-function "arith.ModularSet"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SolveInequalitiesAsCondition
(let [gfn* (delay (jna-base/name->global-function "arith.SolveInequalitiesAsCondition"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SolveInequalitiesDeskewRange
(let [gfn* (delay (jna-base/name->global-function "arith.SolveInequalitiesDeskewRange"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SolveInequalitiesToRange
(let [gfn* (delay (jna-base/name->global-function "arith.SolveInequalitiesToRange"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SolveLinearEquations
(let [gfn* (delay (jna-base/name->global-function "arith.SolveLinearEquations"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} intset_interval
(let [gfn* (delay (jna-base/name->global-function "arith.intset_interval"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} intset_single_point
(let [gfn* (delay (jna-base/name->global-function "arith.intset_single_point"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} intset_vector
(let [gfn* (delay (jna-base/name->global-function "arith.intset_vector"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

