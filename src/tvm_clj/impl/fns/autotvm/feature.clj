(ns tvm-clj.jna.fns.autotvm.feature
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "autotvm.feature.GetCurveSampleFeatureFlatten"))]
  (defn GetCurveSampleFeatureFlatten
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "autotvm.feature.GetCurveSampleFeatureFlatten"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "autotvm.feature.GetItervarFeature"))]
  (defn GetItervarFeature
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "autotvm.feature.GetItervarFeature"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "autotvm.feature.GetItervarFeatureFlatten"))]
  (defn GetItervarFeatureFlatten
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "autotvm.feature.GetItervarFeatureFlatten"}
     (apply jna-base/call-function @gfn* args))))

