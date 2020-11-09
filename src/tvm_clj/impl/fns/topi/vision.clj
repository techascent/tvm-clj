(ns tvm-clj.jna.fns.topi.vision
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.vision.reorg"))]
  (defn reorg
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.vision.reorg"}
     (apply jna-base/call-function @gfn* args))))

