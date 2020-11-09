(ns tvm-clj.impl.fns.schedule
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AutoInlineElemWise-fnptr* (delay (base/name->global-function "schedule.AutoInlineElemWise")))
(defn AutoInlineElemWise
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.AutoInlineElemWise"}
   (apply base/call-function @AutoInlineElemWise-fnptr* args)))

(defonce ^:private AutoInlineInjective-fnptr* (delay (base/name->global-function "schedule.AutoInlineInjective")))
(defn AutoInlineInjective
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.AutoInlineInjective"}
   (apply base/call-function @AutoInlineInjective-fnptr* args)))

(defonce ^:private CreateAttachPath-fnptr* (delay (base/name->global-function "schedule.CreateAttachPath")))
(defn CreateAttachPath
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.CreateAttachPath"}
   (apply base/call-function @CreateAttachPath-fnptr* args)))

(defonce ^:private CreateReadGraph-fnptr* (delay (base/name->global-function "schedule.CreateReadGraph")))
(defn CreateReadGraph
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.CreateReadGraph"}
   (apply base/call-function @CreateReadGraph-fnptr* args)))

(defonce ^:private InferBound-fnptr* (delay (base/name->global-function "schedule.InferBound")))
(defn InferBound
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.InferBound"}
   (apply base/call-function @InferBound-fnptr* args)))

(defonce ^:private PostDFSOrder-fnptr* (delay (base/name->global-function "schedule.PostDFSOrder")))
(defn PostDFSOrder
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.PostDFSOrder"}
   (apply base/call-function @PostDFSOrder-fnptr* args)))

(defonce ^:private ScanFixPointAnalysis-fnptr* (delay (base/name->global-function "schedule.ScanFixPointAnalysis")))
(defn ScanFixPointAnalysis
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.ScanFixPointAnalysis"}
   (apply base/call-function @ScanFixPointAnalysis-fnptr* args)))

(defonce ^:private ScanGetBody-fnptr* (delay (base/name->global-function "schedule.ScanGetBody")))
(defn ScanGetBody
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.ScanGetBody"}
   (apply base/call-function @ScanGetBody-fnptr* args)))

(defonce ^:private ScheduleOps-fnptr* (delay (base/name->global-function "schedule.ScheduleOps")))
(defn ScheduleOps
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.ScheduleOps"}
   (apply base/call-function @ScheduleOps-fnptr* args)))

(defonce ^:private SchedulePostProcRewriteForTensorCore-fnptr* (delay (base/name->global-function "schedule.SchedulePostProcRewriteForTensorCore")))
(defn SchedulePostProcRewriteForTensorCore
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.SchedulePostProcRewriteForTensorCore"}
   (apply base/call-function @SchedulePostProcRewriteForTensorCore-fnptr* args)))

(defonce ^:private SchedulePostProcToPrimFunc-fnptr* (delay (base/name->global-function "schedule.SchedulePostProcToPrimFunc")))
(defn SchedulePostProcToPrimFunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.SchedulePostProcToPrimFunc"}
   (apply base/call-function @SchedulePostProcToPrimFunc-fnptr* args)))

(defonce ^:private VerifyCompactBuffer-fnptr* (delay (base/name->global-function "schedule.VerifyCompactBuffer")))
(defn VerifyCompactBuffer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "schedule.VerifyCompactBuffer"}
   (apply base/call-function @VerifyCompactBuffer-fnptr* args)))

