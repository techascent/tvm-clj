(ns tvm-clj.jna.fns.autotvm.feature
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetCurveSampleFeatureFlatten
(let [gfn* (delay (jna-base/name->global-function "autotvm.feature.GetCurveSampleFeatureFlatten"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetItervarFeature
(let [gfn* (delay (jna-base/name->global-function "autotvm.feature.GetItervarFeature"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GetItervarFeatureFlatten
(let [gfn* (delay (jna-base/name->global-function "autotvm.feature.GetItervarFeatureFlatten"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

