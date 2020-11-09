(ns tvm-clj.impl.fns.topi.vision
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private reorg-fnptr* (delay (base/name->global-function "topi.vision.reorg")))
(defn reorg
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.vision.reorg"}
   (apply base/call-function @reorg-fnptr* args)))

