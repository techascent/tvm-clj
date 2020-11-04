(ns tvm-clj.jna.fns.schedule
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AutoInlineElemWise
(let [gfn* (delay (jna-base/name->global-function "schedule.AutoInlineElemWise"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AutoInlineInjective
(let [gfn* (delay (jna-base/name->global-function "schedule.AutoInlineInjective"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateAttachPath
(let [gfn* (delay (jna-base/name->global-function "schedule.CreateAttachPath"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateReadGraph
(let [gfn* (delay (jna-base/name->global-function "schedule.CreateReadGraph"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} InferBound
(let [gfn* (delay (jna-base/name->global-function "schedule.InferBound"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PostDFSOrder
(let [gfn* (delay (jna-base/name->global-function "schedule.PostDFSOrder"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScanFixPointAnalysis
(let [gfn* (delay (jna-base/name->global-function "schedule.ScanFixPointAnalysis"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScanGetBody
(let [gfn* (delay (jna-base/name->global-function "schedule.ScanGetBody"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ScheduleOps
(let [gfn* (delay (jna-base/name->global-function "schedule.ScheduleOps"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SchedulePostProcRewriteForTensorCore
(let [gfn* (delay (jna-base/name->global-function "schedule.SchedulePostProcRewriteForTensorCore"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SchedulePostProcToPrimFunc
(let [gfn* (delay (jna-base/name->global-function "schedule.SchedulePostProcToPrimFunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} VerifyCompactBuffer
(let [gfn* (delay (jna-base/name->global-function "schedule.VerifyCompactBuffer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

