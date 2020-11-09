(ns tvm-clj.impl.fns.transform
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private EnterPassContext-fnptr* (delay (base/name->global-function "transform.EnterPassContext")))
(defn EnterPassContext
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.EnterPassContext"}
   (apply base/call-function @EnterPassContext-fnptr* args)))

(defonce ^:private ExitPassContext-fnptr* (delay (base/name->global-function "transform.ExitPassContext")))
(defn ExitPassContext
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.ExitPassContext"}
   (apply base/call-function @ExitPassContext-fnptr* args)))

(defonce ^:private GetCurrentPassContext-fnptr* (delay (base/name->global-function "transform.GetCurrentPassContext")))
(defn GetCurrentPassContext
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.GetCurrentPassContext"}
   (apply base/call-function @GetCurrentPassContext-fnptr* args)))

(defonce ^:private Info-fnptr* (delay (base/name->global-function "transform.Info")))
(defn Info
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.Info"}
   (apply base/call-function @Info-fnptr* args)))

(defonce ^:private MakeModulePass-fnptr* (delay (base/name->global-function "transform.MakeModulePass")))
(defn MakeModulePass
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.MakeModulePass"}
   (apply base/call-function @MakeModulePass-fnptr* args)))

(defonce ^:private PassContext-fnptr* (delay (base/name->global-function "transform.PassContext")))
(defn PassContext
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.PassContext"}
   (apply base/call-function @PassContext-fnptr* args)))

(defonce ^:private PassInfo-fnptr* (delay (base/name->global-function "transform.PassInfo")))
(defn PassInfo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.PassInfo"}
   (apply base/call-function @PassInfo-fnptr* args)))

(defonce ^:private PrintIR-fnptr* (delay (base/name->global-function "transform.PrintIR")))
(defn PrintIR
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.PrintIR"}
   (apply base/call-function @PrintIR-fnptr* args)))

(defonce ^:private RunPass-fnptr* (delay (base/name->global-function "transform.RunPass")))
(defn RunPass
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.RunPass"}
   (apply base/call-function @RunPass-fnptr* args)))

(defonce ^:private Sequential-fnptr* (delay (base/name->global-function "transform.Sequential")))
(defn Sequential
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "transform.Sequential"}
   (apply base/call-function @Sequential-fnptr* args)))

