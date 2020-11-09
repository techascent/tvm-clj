(ns tvm-clj.impl.fns.parser
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ParseExpr-fnptr* (delay (base/name->global-function "parser.ParseExpr")))
(defn ParseExpr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "parser.ParseExpr"}
   (apply base/call-function @ParseExpr-fnptr* args)))

(defonce ^:private ParseModule-fnptr* (delay (base/name->global-function "parser.ParseModule")))
(defn ParseModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "parser.ParseModule"}
   (apply base/call-function @ParseModule-fnptr* args)))

