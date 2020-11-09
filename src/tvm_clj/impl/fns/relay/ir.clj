(ns tvm-clj.impl.fns.relay.ir
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private Any-fnptr* (delay (base/name->global-function "relay.ir.Any")))
(defn Any
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Any"}
   (apply base/call-function @Any-fnptr* args)))

(defonce ^:private Bind-fnptr* (delay (base/name->global-function "relay.ir.Bind")))
(defn Bind
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Bind"}
   (apply base/call-function @Bind-fnptr* args)))

(defonce ^:private Call-fnptr* (delay (base/name->global-function "relay.ir.Call")))
(defn Call
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Call"}
   (apply base/call-function @Call-fnptr* args)))

(defonce ^:private Clause-fnptr* (delay (base/name->global-function "relay.ir.Clause")))
(defn Clause
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Clause"}
   (apply base/call-function @Clause-fnptr* args)))

(defonce ^:private Constant-fnptr* (delay (base/name->global-function "relay.ir.Constant")))
(defn Constant
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Constant"}
   (apply base/call-function @Constant-fnptr* args)))

(defonce ^:private Function-fnptr* (delay (base/name->global-function "relay.ir.Function")))
(defn Function
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Function"}
   (apply base/call-function @Function-fnptr* args)))

(defonce ^:private If-fnptr* (delay (base/name->global-function "relay.ir.If")))
(defn If
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.If"}
   (apply base/call-function @If-fnptr* args)))

(defonce ^:private IsDynamic-fnptr* (delay (base/name->global-function "relay.ir.IsDynamic")))
(defn IsDynamic
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.IsDynamic"}
   (apply base/call-function @IsDynamic-fnptr* args)))

(defonce ^:private Let-fnptr* (delay (base/name->global-function "relay.ir.Let")))
(defn Let
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Let"}
   (apply base/call-function @Let-fnptr* args)))

(defonce ^:private Match-fnptr* (delay (base/name->global-function "relay.ir.Match")))
(defn Match
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Match"}
   (apply base/call-function @Match-fnptr* args)))

(defonce ^:private PatternConstructor-fnptr* (delay (base/name->global-function "relay.ir.PatternConstructor")))
(defn PatternConstructor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.PatternConstructor"}
   (apply base/call-function @PatternConstructor-fnptr* args)))

(defonce ^:private PatternTuple-fnptr* (delay (base/name->global-function "relay.ir.PatternTuple")))
(defn PatternTuple
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.PatternTuple"}
   (apply base/call-function @PatternTuple-fnptr* args)))

(defonce ^:private PatternVar-fnptr* (delay (base/name->global-function "relay.ir.PatternVar")))
(defn PatternVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.PatternVar"}
   (apply base/call-function @PatternVar-fnptr* args)))

(defonce ^:private PatternWildcard-fnptr* (delay (base/name->global-function "relay.ir.PatternWildcard")))
(defn PatternWildcard
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.PatternWildcard"}
   (apply base/call-function @PatternWildcard-fnptr* args)))

(defonce ^:private RefCreate-fnptr* (delay (base/name->global-function "relay.ir.RefCreate")))
(defn RefCreate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.RefCreate"}
   (apply base/call-function @RefCreate-fnptr* args)))

(defonce ^:private RefRead-fnptr* (delay (base/name->global-function "relay.ir.RefRead")))
(defn RefRead
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.RefRead"}
   (apply base/call-function @RefRead-fnptr* args)))

(defonce ^:private RefWrite-fnptr* (delay (base/name->global-function "relay.ir.RefWrite")))
(defn RefWrite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.RefWrite"}
   (apply base/call-function @RefWrite-fnptr* args)))

(defonce ^:private TempExprRealize-fnptr* (delay (base/name->global-function "relay.ir.TempExprRealize")))
(defn TempExprRealize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.TempExprRealize"}
   (apply base/call-function @TempExprRealize-fnptr* args)))

(defonce ^:private Tuple-fnptr* (delay (base/name->global-function "relay.ir.Tuple")))
(defn Tuple
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Tuple"}
   (apply base/call-function @Tuple-fnptr* args)))

(defonce ^:private TupleGetItem-fnptr* (delay (base/name->global-function "relay.ir.TupleGetItem")))
(defn TupleGetItem
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.TupleGetItem"}
   (apply base/call-function @TupleGetItem-fnptr* args)))

(defonce ^:private Var-fnptr* (delay (base/name->global-function "relay.ir.Var")))
(defn Var
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.Var"}
   (apply base/call-function @Var-fnptr* args)))

(defonce ^:private cast-fnptr* (delay (base/name->global-function "relay.ir.cast")))
(defn cast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.cast"}
   (apply base/call-function @cast-fnptr* args)))

(defonce ^:private cast_like-fnptr* (delay (base/name->global-function "relay.ir.cast_like")))
(defn cast_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.ir.cast_like"}
   (apply base/call-function @cast_like-fnptr* args)))

