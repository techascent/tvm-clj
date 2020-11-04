(ns tvm-clj.jna.fns.test.op
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "test.op.InferTensorizeRegion"))]
  (defn InferTensorizeRegion
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "test.op.InferTensorizeRegion"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "test.op.MatchTensorizeBody"))]
  (defn MatchTensorizeBody
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "test.op.MatchTensorizeBody"}
     (apply jna-base/call-function @gfn* args))))

