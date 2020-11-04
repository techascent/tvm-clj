(ns tvm-clj.jna.fns.transform
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EnterPassContext
(let [gfn* (delay (jna-base/name->global-function "transform.EnterPassContext"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ExitPassContext
(let [gfn* (delay (jna-base/name->global-function "transform.ExitPassContext"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetCurrentPassContext
(let [gfn* (delay (jna-base/name->global-function "transform.GetCurrentPassContext"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Info
(let [gfn* (delay (jna-base/name->global-function "transform.Info"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MakeModulePass
(let [gfn* (delay (jna-base/name->global-function "transform.MakeModulePass"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PassContext
(let [gfn* (delay (jna-base/name->global-function "transform.PassContext"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PassInfo
(let [gfn* (delay (jna-base/name->global-function "transform.PassInfo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PrintIR
(let [gfn* (delay (jna-base/name->global-function "transform.PrintIR"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RunPass
(let [gfn* (delay (jna-base/name->global-function "transform.RunPass"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Sequential
(let [gfn* (delay (jna-base/name->global-function "transform.Sequential"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

