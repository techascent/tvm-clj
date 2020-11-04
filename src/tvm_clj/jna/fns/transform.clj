(ns tvm-clj.jna.fns.transform
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "transform.EnterPassContext"))]
  (defn EnterPassContext
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.EnterPassContext"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.ExitPassContext"))]
  (defn ExitPassContext
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.ExitPassContext"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.GetCurrentPassContext"))]
  (defn GetCurrentPassContext
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.GetCurrentPassContext"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.Info"))]
  (defn Info
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.Info"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.MakeModulePass"))]
  (defn MakeModulePass
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.MakeModulePass"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.PassContext"))]
  (defn PassContext
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.PassContext"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.PassInfo"))]
  (defn PassInfo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.PassInfo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.PrintIR"))]
  (defn PrintIR
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.PrintIR"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.RunPass"))]
  (defn RunPass
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.RunPass"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "transform.Sequential"))]
  (defn Sequential
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "transform.Sequential"}
     (apply jna-base/call-function @gfn* args))))

