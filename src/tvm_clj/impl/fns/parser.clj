(ns tvm-clj.jna.fns.parser
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "parser.ParseExpr"))]
  (defn ParseExpr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "parser.ParseExpr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "parser.ParseModule"))]
  (defn ParseModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "parser.ParseModule"}
     (apply jna-base/call-function @gfn* args))))

