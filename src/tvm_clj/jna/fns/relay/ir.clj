(ns tvm-clj.jna.fns.relay.ir
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Any"))]
  (defn Any
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Any"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Bind"))]
  (defn Bind
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Bind"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Call"))]
  (defn Call
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Call"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Clause"))]
  (defn Clause
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Clause"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Constant"))]
  (defn Constant
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Constant"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Function"))]
  (defn Function
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Function"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.If"))]
  (defn If
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.If"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.IsDynamic"))]
  (defn IsDynamic
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.IsDynamic"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Let"))]
  (defn Let
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Let"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Match"))]
  (defn Match
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Match"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternConstructor"))]
  (defn PatternConstructor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.PatternConstructor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternTuple"))]
  (defn PatternTuple
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.PatternTuple"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternVar"))]
  (defn PatternVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.PatternVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.PatternWildcard"))]
  (defn PatternWildcard
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.PatternWildcard"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.RefCreate"))]
  (defn RefCreate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.RefCreate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.RefRead"))]
  (defn RefRead
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.RefRead"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.RefWrite"))]
  (defn RefWrite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.RefWrite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.TempExprRealize"))]
  (defn TempExprRealize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.TempExprRealize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Tuple"))]
  (defn Tuple
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Tuple"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.TupleGetItem"))]
  (defn TupleGetItem
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.TupleGetItem"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.Var"))]
  (defn Var
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.Var"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.cast"))]
  (defn cast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.cast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.ir.cast_like"))]
  (defn cast_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.ir.cast_like"}
     (apply jna-base/call-function @gfn* args))))

