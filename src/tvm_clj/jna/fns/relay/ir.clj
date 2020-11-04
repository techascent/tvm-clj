(ns tvm-clj.jna.fns.relay.ir
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Any
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Any"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Bind
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Bind"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Call
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Call"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Clause
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Clause"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Constant
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Constant"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Function
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Function"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} If
(let [gfn* (delay (jna-base/name->global-function "relay.ir.If"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IsDynamic
(let [gfn* (delay (jna-base/name->global-function "relay.ir.IsDynamic"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Let
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Let"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Match
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Match"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PatternConstructor
(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternConstructor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PatternTuple
(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternTuple"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PatternVar
(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PatternWildcard
(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternWildcard"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RefCreate
(let [gfn* (delay (jna-base/name->global-function "relay.ir.RefCreate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RefRead
(let [gfn* (delay (jna-base/name->global-function "relay.ir.RefRead"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} RefWrite
(let [gfn* (delay (jna-base/name->global-function "relay.ir.RefWrite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TempExprRealize
(let [gfn* (delay (jna-base/name->global-function "relay.ir.TempExprRealize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Tuple
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Tuple"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TupleGetItem
(let [gfn* (delay (jna-base/name->global-function "relay.ir.TupleGetItem"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Var
(let [gfn* (delay (jna-base/name->global-function "relay.ir.Var"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cast
(let [gfn* (delay (jna-base/name->global-function "relay.ir.cast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cast_like
(let [gfn* (delay (jna-base/name->global-function "relay.ir.cast_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

