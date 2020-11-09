(ns tvm-clj.impl.fns.support
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private GetLibInfo-fnptr* (delay (base/name->global-function "support.GetLibInfo")))
(defn GetLibInfo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "support.GetLibInfo"}
   (apply base/call-function @GetLibInfo-fnptr* args)))

