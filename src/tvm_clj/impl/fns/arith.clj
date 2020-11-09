(ns tvm-clj.impl.fns.arith
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ConstIntBound-fnptr* (delay (base/name->global-function "arith.ConstIntBound")))
(defn ConstIntBound
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.ConstIntBound"}
   (apply base/call-function @ConstIntBound-fnptr* args)))

(defonce ^:private CreateAnalyzer-fnptr* (delay (base/name->global-function "arith.CreateAnalyzer")))
(defn CreateAnalyzer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.CreateAnalyzer"}
   (apply base/call-function @CreateAnalyzer-fnptr* args)))

(defonce ^:private DeduceBound-fnptr* (delay (base/name->global-function "arith.DeduceBound")))
(defn DeduceBound
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.DeduceBound"}
   (apply base/call-function @DeduceBound-fnptr* args)))

(defonce ^:private DetectClipBound-fnptr* (delay (base/name->global-function "arith.DetectClipBound")))
(defn DetectClipBound
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.DetectClipBound"}
   (apply base/call-function @DetectClipBound-fnptr* args)))

(defonce ^:private DetectLinearEquation-fnptr* (delay (base/name->global-function "arith.DetectLinearEquation")))
(defn DetectLinearEquation
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.DetectLinearEquation"}
   (apply base/call-function @DetectLinearEquation-fnptr* args)))

(defonce ^:private DomainTouched-fnptr* (delay (base/name->global-function "arith.DomainTouched")))
(defn DomainTouched
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.DomainTouched"}
   (apply base/call-function @DomainTouched-fnptr* args)))

(defonce ^:private IntConstraints-fnptr* (delay (base/name->global-function "arith.IntConstraints")))
(defn IntConstraints
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntConstraints"}
   (apply base/call-function @IntConstraints-fnptr* args)))

(defonce ^:private IntConstraintsTransform-fnptr* (delay (base/name->global-function "arith.IntConstraintsTransform")))
(defn IntConstraintsTransform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntConstraintsTransform"}
   (apply base/call-function @IntConstraintsTransform-fnptr* args)))

(defonce ^:private IntGroupBounds-fnptr* (delay (base/name->global-function "arith.IntGroupBounds")))
(defn IntGroupBounds
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntGroupBounds"}
   (apply base/call-function @IntGroupBounds-fnptr* args)))

(defonce ^:private IntGroupBounds_FindBestRange-fnptr* (delay (base/name->global-function "arith.IntGroupBounds_FindBestRange")))
(defn IntGroupBounds_FindBestRange
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntGroupBounds_FindBestRange"}
   (apply base/call-function @IntGroupBounds_FindBestRange-fnptr* args)))

(defonce ^:private IntGroupBounds_from_range-fnptr* (delay (base/name->global-function "arith.IntGroupBounds_from_range")))
(defn IntGroupBounds_from_range
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntGroupBounds_from_range"}
   (apply base/call-function @IntGroupBounds_from_range-fnptr* args)))

(defonce ^:private IntSetIsEverything-fnptr* (delay (base/name->global-function "arith.IntSetIsEverything")))
(defn IntSetIsEverything
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntSetIsEverything"}
   (apply base/call-function @IntSetIsEverything-fnptr* args)))

(defonce ^:private IntSetIsNothing-fnptr* (delay (base/name->global-function "arith.IntSetIsNothing")))
(defn IntSetIsNothing
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntSetIsNothing"}
   (apply base/call-function @IntSetIsNothing-fnptr* args)))

(defonce ^:private IntervalSet-fnptr* (delay (base/name->global-function "arith.IntervalSet")))
(defn IntervalSet
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntervalSet"}
   (apply base/call-function @IntervalSet-fnptr* args)))

(defonce ^:private IntervalSetGetMax-fnptr* (delay (base/name->global-function "arith.IntervalSetGetMax")))
(defn IntervalSetGetMax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntervalSetGetMax"}
   (apply base/call-function @IntervalSetGetMax-fnptr* args)))

(defonce ^:private IntervalSetGetMin-fnptr* (delay (base/name->global-function "arith.IntervalSetGetMin")))
(defn IntervalSetGetMin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.IntervalSetGetMin"}
   (apply base/call-function @IntervalSetGetMin-fnptr* args)))

(defonce ^:private ModularSet-fnptr* (delay (base/name->global-function "arith.ModularSet")))
(defn ModularSet
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.ModularSet"}
   (apply base/call-function @ModularSet-fnptr* args)))

(defonce ^:private SolveInequalitiesAsCondition-fnptr* (delay (base/name->global-function "arith.SolveInequalitiesAsCondition")))
(defn SolveInequalitiesAsCondition
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.SolveInequalitiesAsCondition"}
   (apply base/call-function @SolveInequalitiesAsCondition-fnptr* args)))

(defonce ^:private SolveInequalitiesDeskewRange-fnptr* (delay (base/name->global-function "arith.SolveInequalitiesDeskewRange")))
(defn SolveInequalitiesDeskewRange
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.SolveInequalitiesDeskewRange"}
   (apply base/call-function @SolveInequalitiesDeskewRange-fnptr* args)))

(defonce ^:private SolveInequalitiesToRange-fnptr* (delay (base/name->global-function "arith.SolveInequalitiesToRange")))
(defn SolveInequalitiesToRange
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.SolveInequalitiesToRange"}
   (apply base/call-function @SolveInequalitiesToRange-fnptr* args)))

(defonce ^:private SolveLinearEquations-fnptr* (delay (base/name->global-function "arith.SolveLinearEquations")))
(defn SolveLinearEquations
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.SolveLinearEquations"}
   (apply base/call-function @SolveLinearEquations-fnptr* args)))

(defonce ^:private intset_interval-fnptr* (delay (base/name->global-function "arith.intset_interval")))
(defn intset_interval
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.intset_interval"}
   (apply base/call-function @intset_interval-fnptr* args)))

(defonce ^:private intset_single_point-fnptr* (delay (base/name->global-function "arith.intset_single_point")))
(defn intset_single_point
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.intset_single_point"}
   (apply base/call-function @intset_single_point-fnptr* args)))

(defonce ^:private intset_vector-fnptr* (delay (base/name->global-function "arith.intset_vector")))
(defn intset_vector
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "arith.intset_vector"}
   (apply base/call-function @intset_vector-fnptr* args)))

