(ns tvm-clj.jna.fns.arith
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "arith.ConstIntBound"))]
  (defn ConstIntBound
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.ConstIntBound"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.CreateAnalyzer"))]
  (defn CreateAnalyzer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.CreateAnalyzer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.DeduceBound"))]
  (defn DeduceBound
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.DeduceBound"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.DetectClipBound"))]
  (defn DetectClipBound
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.DetectClipBound"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.DetectLinearEquation"))]
  (defn DetectLinearEquation
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.DetectLinearEquation"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.DomainTouched"))]
  (defn DomainTouched
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.DomainTouched"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntConstraints"))]
  (defn IntConstraints
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntConstraints"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntConstraintsTransform"))]
  (defn IntConstraintsTransform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntConstraintsTransform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntGroupBounds"))]
  (defn IntGroupBounds
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntGroupBounds"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntGroupBounds_FindBestRange"))]
  (defn IntGroupBounds_FindBestRange
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntGroupBounds_FindBestRange"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntGroupBounds_from_range"))]
  (defn IntGroupBounds_from_range
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntGroupBounds_from_range"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntSetIsEverything"))]
  (defn IntSetIsEverything
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntSetIsEverything"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntSetIsNothing"))]
  (defn IntSetIsNothing
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntSetIsNothing"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntervalSet"))]
  (defn IntervalSet
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntervalSet"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntervalSetGetMax"))]
  (defn IntervalSetGetMax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntervalSetGetMax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.IntervalSetGetMin"))]
  (defn IntervalSetGetMin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.IntervalSetGetMin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.ModularSet"))]
  (defn ModularSet
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.ModularSet"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.SolveInequalitiesAsCondition"))]
  (defn SolveInequalitiesAsCondition
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.SolveInequalitiesAsCondition"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.SolveInequalitiesDeskewRange"))]
  (defn SolveInequalitiesDeskewRange
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.SolveInequalitiesDeskewRange"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.SolveInequalitiesToRange"))]
  (defn SolveInequalitiesToRange
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.SolveInequalitiesToRange"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.SolveLinearEquations"))]
  (defn SolveLinearEquations
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.SolveLinearEquations"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.intset_interval"))]
  (defn intset_interval
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.intset_interval"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.intset_single_point"))]
  (defn intset_single_point
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.intset_single_point"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "arith.intset_vector"))]
  (defn intset_vector
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "arith.intset_vector"}
     (apply jna-base/call-function @gfn* args))))

