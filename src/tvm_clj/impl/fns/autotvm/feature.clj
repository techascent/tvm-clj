(ns tvm-clj.impl.fns.autotvm.feature
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private GetCurveSampleFeatureFlatten-fnptr* (delay (base/name->global-function "autotvm.feature.GetCurveSampleFeatureFlatten")))
(defn GetCurveSampleFeatureFlatten
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "autotvm.feature.GetCurveSampleFeatureFlatten"}
   (apply base/call-function @GetCurveSampleFeatureFlatten-fnptr* args)))

(defonce ^:private GetItervarFeature-fnptr* (delay (base/name->global-function "autotvm.feature.GetItervarFeature")))
(defn GetItervarFeature
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "autotvm.feature.GetItervarFeature"}
   (apply base/call-function @GetItervarFeature-fnptr* args)))

(defonce ^:private GetItervarFeatureFlatten-fnptr* (delay (base/name->global-function "autotvm.feature.GetItervarFeatureFlatten")))
(defn GetItervarFeatureFlatten
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "autotvm.feature.GetItervarFeatureFlatten"}
   (apply base/call-function @GetItervarFeatureFlatten-fnptr* args)))

