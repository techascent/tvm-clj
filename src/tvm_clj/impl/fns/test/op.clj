(ns tvm-clj.impl.fns.test.op
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private InferTensorizeRegion-fnptr* (delay (base/name->global-function "test.op.InferTensorizeRegion")))
(defn InferTensorizeRegion
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "test.op.InferTensorizeRegion"}
   (apply base/call-function @InferTensorizeRegion-fnptr* args)))

(defonce ^:private MatchTensorizeBody-fnptr* (delay (base/name->global-function "test.op.MatchTensorizeBody")))
(defn MatchTensorizeBody
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "test.op.MatchTensorizeBody"}
   (apply base/call-function @MatchTensorizeBody-fnptr* args)))

