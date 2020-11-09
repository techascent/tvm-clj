(ns tvm-clj.jna.fns.relay.analysis
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.AnnotatedRegionSet"))]
  (defn AnnotatedRegionSet
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.AnnotatedRegionSet"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.CallGraph"))]
  (defn CallGraph
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.CallGraph"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.CollectDeviceAnnotationOps"))]
  (defn CollectDeviceAnnotationOps
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.CollectDeviceAnnotationOps"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.CollectDeviceInfo"))]
  (defn CollectDeviceInfo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.CollectDeviceInfo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.ContextAnalysis"))]
  (defn ContextAnalysis
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.ContextAnalysis"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.ExtractFusedFunctions"))]
  (defn ExtractFusedFunctions
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.ExtractFusedFunctions"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetGlobalVarCallCount"))]
  (defn GetGlobalVarCallCount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.GetGlobalVarCallCount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetModule"))]
  (defn GetModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.GetModule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetRefCountGlobalVar"))]
  (defn GetRefCountGlobalVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.GetRefCountGlobalVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetRegion"))]
  (defn GetRegion
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.GetRegion"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.GetTotalMacNumber"))]
  (defn GetTotalMacNumber
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.GetTotalMacNumber"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.IsRecursive"))]
  (defn IsRecursive
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.IsRecursive"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.PrintCallGraph"))]
  (defn PrintCallGraph
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.PrintCallGraph"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.PrintCallGraphGlobalVar"))]
  (defn PrintCallGraphGlobalVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.PrintCallGraphGlobalVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis._test_type_solver"))]
  (defn _test_type_solver
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis._test_type_solver"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.all_dtypes"))]
  (defn all_dtypes
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.all_dtypes"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.all_type_vars"))]
  (defn all_type_vars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.all_type_vars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.all_vars"))]
  (defn all_vars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.all_vars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.bound_type_vars"))]
  (defn bound_type_vars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.bound_type_vars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.bound_vars"))]
  (defn bound_vars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.bound_vars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.check_basic_block_normal_form"))]
  (defn check_basic_block_normal_form
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.check_basic_block_normal_form"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.check_constant"))]
  (defn check_constant
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.check_constant"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.check_kind"))]
  (defn check_kind
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.check_kind"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.detect_feature"))]
  (defn detect_feature
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.detect_feature"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.free_type_vars"))]
  (defn free_type_vars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.free_type_vars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.free_vars"))]
  (defn free_vars
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.free_vars"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.get_calibrate_module"))]
  (defn get_calibrate_module
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.get_calibrate_module"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.get_calibrate_output_map"))]
  (defn get_calibrate_output_map
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.get_calibrate_output_map"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.post_order_visit"))]
  (defn post_order_visit
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.post_order_visit"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.search_dense_op_weight"))]
  (defn search_dense_op_weight
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.search_dense_op_weight"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.search_fc_transpose"))]
  (defn search_fc_transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.search_fc_transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.unmatched_cases"))]
  (defn unmatched_cases
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.unmatched_cases"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.analysis.well_formed"))]
  (defn well_formed
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.analysis.well_formed"}
     (apply jna-base/call-function @gfn* args))))

