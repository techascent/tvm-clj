(ns tvm-clj.impl.fns.relay.analysis
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AnnotatedRegionSet-fnptr* (delay (base/name->global-function "relay.analysis.AnnotatedRegionSet")))
(defn AnnotatedRegionSet
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.AnnotatedRegionSet"}
   (apply base/call-function @AnnotatedRegionSet-fnptr* args)))

(defonce ^:private CallGraph-fnptr* (delay (base/name->global-function "relay.analysis.CallGraph")))
(defn CallGraph
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.CallGraph"}
   (apply base/call-function @CallGraph-fnptr* args)))

(defonce ^:private CollectDeviceAnnotationOps-fnptr* (delay (base/name->global-function "relay.analysis.CollectDeviceAnnotationOps")))
(defn CollectDeviceAnnotationOps
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.CollectDeviceAnnotationOps"}
   (apply base/call-function @CollectDeviceAnnotationOps-fnptr* args)))

(defonce ^:private CollectDeviceInfo-fnptr* (delay (base/name->global-function "relay.analysis.CollectDeviceInfo")))
(defn CollectDeviceInfo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.CollectDeviceInfo"}
   (apply base/call-function @CollectDeviceInfo-fnptr* args)))

(defonce ^:private ContextAnalysis-fnptr* (delay (base/name->global-function "relay.analysis.ContextAnalysis")))
(defn ContextAnalysis
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.ContextAnalysis"}
   (apply base/call-function @ContextAnalysis-fnptr* args)))

(defonce ^:private ExtractFusedFunctions-fnptr* (delay (base/name->global-function "relay.analysis.ExtractFusedFunctions")))
(defn ExtractFusedFunctions
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.ExtractFusedFunctions"}
   (apply base/call-function @ExtractFusedFunctions-fnptr* args)))

(defonce ^:private GetGlobalVarCallCount-fnptr* (delay (base/name->global-function "relay.analysis.GetGlobalVarCallCount")))
(defn GetGlobalVarCallCount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.GetGlobalVarCallCount"}
   (apply base/call-function @GetGlobalVarCallCount-fnptr* args)))

(defonce ^:private GetModule-fnptr* (delay (base/name->global-function "relay.analysis.GetModule")))
(defn GetModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.GetModule"}
   (apply base/call-function @GetModule-fnptr* args)))

(defonce ^:private GetRefCountGlobalVar-fnptr* (delay (base/name->global-function "relay.analysis.GetRefCountGlobalVar")))
(defn GetRefCountGlobalVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.GetRefCountGlobalVar"}
   (apply base/call-function @GetRefCountGlobalVar-fnptr* args)))

(defonce ^:private GetRegion-fnptr* (delay (base/name->global-function "relay.analysis.GetRegion")))
(defn GetRegion
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.GetRegion"}
   (apply base/call-function @GetRegion-fnptr* args)))

(defonce ^:private GetTotalMacNumber-fnptr* (delay (base/name->global-function "relay.analysis.GetTotalMacNumber")))
(defn GetTotalMacNumber
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.GetTotalMacNumber"}
   (apply base/call-function @GetTotalMacNumber-fnptr* args)))

(defonce ^:private IsRecursive-fnptr* (delay (base/name->global-function "relay.analysis.IsRecursive")))
(defn IsRecursive
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.IsRecursive"}
   (apply base/call-function @IsRecursive-fnptr* args)))

(defonce ^:private PrintCallGraph-fnptr* (delay (base/name->global-function "relay.analysis.PrintCallGraph")))
(defn PrintCallGraph
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.PrintCallGraph"}
   (apply base/call-function @PrintCallGraph-fnptr* args)))

(defonce ^:private PrintCallGraphGlobalVar-fnptr* (delay (base/name->global-function "relay.analysis.PrintCallGraphGlobalVar")))
(defn PrintCallGraphGlobalVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.PrintCallGraphGlobalVar"}
   (apply base/call-function @PrintCallGraphGlobalVar-fnptr* args)))

(defonce ^:private _test_type_solver-fnptr* (delay (base/name->global-function "relay.analysis._test_type_solver")))
(defn _test_type_solver
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis._test_type_solver"}
   (apply base/call-function @_test_type_solver-fnptr* args)))

(defonce ^:private all_dtypes-fnptr* (delay (base/name->global-function "relay.analysis.all_dtypes")))
(defn all_dtypes
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.all_dtypes"}
   (apply base/call-function @all_dtypes-fnptr* args)))

(defonce ^:private all_type_vars-fnptr* (delay (base/name->global-function "relay.analysis.all_type_vars")))
(defn all_type_vars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.all_type_vars"}
   (apply base/call-function @all_type_vars-fnptr* args)))

(defonce ^:private all_vars-fnptr* (delay (base/name->global-function "relay.analysis.all_vars")))
(defn all_vars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.all_vars"}
   (apply base/call-function @all_vars-fnptr* args)))

(defonce ^:private bound_type_vars-fnptr* (delay (base/name->global-function "relay.analysis.bound_type_vars")))
(defn bound_type_vars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.bound_type_vars"}
   (apply base/call-function @bound_type_vars-fnptr* args)))

(defonce ^:private bound_vars-fnptr* (delay (base/name->global-function "relay.analysis.bound_vars")))
(defn bound_vars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.bound_vars"}
   (apply base/call-function @bound_vars-fnptr* args)))

(defonce ^:private check_basic_block_normal_form-fnptr* (delay (base/name->global-function "relay.analysis.check_basic_block_normal_form")))
(defn check_basic_block_normal_form
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.check_basic_block_normal_form"}
   (apply base/call-function @check_basic_block_normal_form-fnptr* args)))

(defonce ^:private check_constant-fnptr* (delay (base/name->global-function "relay.analysis.check_constant")))
(defn check_constant
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.check_constant"}
   (apply base/call-function @check_constant-fnptr* args)))

(defonce ^:private check_kind-fnptr* (delay (base/name->global-function "relay.analysis.check_kind")))
(defn check_kind
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.check_kind"}
   (apply base/call-function @check_kind-fnptr* args)))

(defonce ^:private detect_feature-fnptr* (delay (base/name->global-function "relay.analysis.detect_feature")))
(defn detect_feature
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.detect_feature"}
   (apply base/call-function @detect_feature-fnptr* args)))

(defonce ^:private free_type_vars-fnptr* (delay (base/name->global-function "relay.analysis.free_type_vars")))
(defn free_type_vars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.free_type_vars"}
   (apply base/call-function @free_type_vars-fnptr* args)))

(defonce ^:private free_vars-fnptr* (delay (base/name->global-function "relay.analysis.free_vars")))
(defn free_vars
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.free_vars"}
   (apply base/call-function @free_vars-fnptr* args)))

(defonce ^:private get_calibrate_module-fnptr* (delay (base/name->global-function "relay.analysis.get_calibrate_module")))
(defn get_calibrate_module
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.get_calibrate_module"}
   (apply base/call-function @get_calibrate_module-fnptr* args)))

(defonce ^:private get_calibrate_output_map-fnptr* (delay (base/name->global-function "relay.analysis.get_calibrate_output_map")))
(defn get_calibrate_output_map
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.get_calibrate_output_map"}
   (apply base/call-function @get_calibrate_output_map-fnptr* args)))

(defonce ^:private post_order_visit-fnptr* (delay (base/name->global-function "relay.analysis.post_order_visit")))
(defn post_order_visit
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.post_order_visit"}
   (apply base/call-function @post_order_visit-fnptr* args)))

(defonce ^:private search_dense_op_weight-fnptr* (delay (base/name->global-function "relay.analysis.search_dense_op_weight")))
(defn search_dense_op_weight
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.search_dense_op_weight"}
   (apply base/call-function @search_dense_op_weight-fnptr* args)))

(defonce ^:private search_fc_transpose-fnptr* (delay (base/name->global-function "relay.analysis.search_fc_transpose")))
(defn search_fc_transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.search_fc_transpose"}
   (apply base/call-function @search_fc_transpose-fnptr* args)))

(defonce ^:private unmatched_cases-fnptr* (delay (base/name->global-function "relay.analysis.unmatched_cases")))
(defn unmatched_cases
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.unmatched_cases"}
   (apply base/call-function @unmatched_cases-fnptr* args)))

(defonce ^:private well_formed-fnptr* (delay (base/name->global-function "relay.analysis.well_formed")))
(defn well_formed
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.analysis.well_formed"}
   (apply base/call-function @well_formed-fnptr* args)))

