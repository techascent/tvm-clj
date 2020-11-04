(ns tvm-clj.jna.fns.schedule
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "schedule.AutoInlineElemWise"))]
  (defn AutoInlineElemWise
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.AutoInlineElemWise"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.AutoInlineInjective"))]
  (defn AutoInlineInjective
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.AutoInlineInjective"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.CreateAttachPath"))]
  (defn CreateAttachPath
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.CreateAttachPath"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.CreateReadGraph"))]
  (defn CreateReadGraph
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.CreateReadGraph"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.InferBound"))]
  (defn InferBound
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.InferBound"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.PostDFSOrder"))]
  (defn PostDFSOrder
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.PostDFSOrder"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.ScanFixPointAnalysis"))]
  (defn ScanFixPointAnalysis
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.ScanFixPointAnalysis"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.ScanGetBody"))]
  (defn ScanGetBody
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.ScanGetBody"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.ScheduleOps"))]
  (defn ScheduleOps
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.ScheduleOps"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.SchedulePostProcRewriteForTensorCore"))]
  (defn SchedulePostProcRewriteForTensorCore
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.SchedulePostProcRewriteForTensorCore"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.SchedulePostProcToPrimFunc"))]
  (defn SchedulePostProcToPrimFunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.SchedulePostProcToPrimFunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "schedule.VerifyCompactBuffer"))]
  (defn VerifyCompactBuffer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "schedule.VerifyCompactBuffer"}
     (apply jna-base/call-function @gfn* args))))

