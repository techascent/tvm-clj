(ns tvm-clj.jna.fns.support
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "support.GetLibInfo"))]
  (defn GetLibInfo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "support.GetLibInfo"}
     (apply jna-base/call-function @gfn* args))))

