(ns tvm-clj.jna.fns.relay.analysis
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AnnotatedRegionSet
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.AnnotatedRegionSet"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CallGraph
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.CallGraph"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CollectDeviceAnnotationOps
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.CollectDeviceAnnotationOps"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CollectDeviceInfo
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.CollectDeviceInfo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ContextAnalysis
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.ContextAnalysis"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ExtractFusedFunctions
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.ExtractFusedFunctions"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetGlobalVarCallCount
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetGlobalVarCallCount"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetModule
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetRefCountGlobalVar
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetRefCountGlobalVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetRegion
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetRegion"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetTotalMacNumber
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetTotalMacNumber"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IsRecursive
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.IsRecursive"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PrintCallGraph
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.PrintCallGraph"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PrintCallGraphGlobalVar
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.PrintCallGraphGlobalVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _test_type_solver
(let [gfn* (delay (jna-base/name->global-function "relay.analysis._test_type_solver"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} all_dtypes
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.all_dtypes"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} all_type_vars
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.all_type_vars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} all_vars
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.all_vars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bound_type_vars
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.bound_type_vars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bound_vars
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.bound_vars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} check_basic_block_normal_form
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.check_basic_block_normal_form"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} check_constant
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.check_constant"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} check_kind
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.check_kind"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} detect_feature
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.detect_feature"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} free_type_vars
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.free_type_vars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} free_vars
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.free_vars"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} get_calibrate_module
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.get_calibrate_module"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} get_calibrate_output_map
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.get_calibrate_output_map"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} post_order_visit
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.post_order_visit"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} search_dense_op_weight
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.search_dense_op_weight"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} search_fc_transpose
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.search_fc_transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} unmatched_cases
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.unmatched_cases"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} well_formed
(let [gfn* (delay (jna-base/name->global-function "relay.analysis.well_formed"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

