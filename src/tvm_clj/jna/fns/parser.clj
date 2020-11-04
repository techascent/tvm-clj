(ns tvm-clj.jna.fns.parser
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ParseExpr
(let [gfn* (delay (jna-base/name->global-function "parser.ParseExpr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ParseModule
(let [gfn* (delay (jna-base/name->global-function "parser.ParseModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

