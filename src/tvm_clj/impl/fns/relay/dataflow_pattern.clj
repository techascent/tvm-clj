(ns tvm-clj.impl.fns.relay.dataflow_pattern
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private AltPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.AltPattern")))
(defn AltPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.AltPattern"}
   (apply base/call-function @AltPattern-fnptr* args)))

(defonce ^:private AttrPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.AttrPattern")))
(defn AttrPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.AttrPattern"}
   (apply base/call-function @AttrPattern-fnptr* args)))

(defonce ^:private CallPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.CallPattern")))
(defn CallPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.CallPattern"}
   (apply base/call-function @CallPattern-fnptr* args)))

(defonce ^:private ConstantPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.ConstantPattern")))
(defn ConstantPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.ConstantPattern"}
   (apply base/call-function @ConstantPattern-fnptr* args)))

(defonce ^:private DFPatternCallback-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.DFPatternCallback")))
(defn DFPatternCallback
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.DFPatternCallback"}
   (apply base/call-function @DFPatternCallback-fnptr* args)))

(defonce ^:private DataTypePattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.DataTypePattern")))
(defn DataTypePattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.DataTypePattern"}
   (apply base/call-function @DataTypePattern-fnptr* args)))

(defonce ^:private DominatorPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.DominatorPattern")))
(defn DominatorPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.DominatorPattern"}
   (apply base/call-function @DominatorPattern-fnptr* args)))

(defonce ^:private ExprPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.ExprPattern")))
(defn ExprPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.ExprPattern"}
   (apply base/call-function @ExprPattern-fnptr* args)))

(defonce ^:private ShapePattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.ShapePattern")))
(defn ShapePattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.ShapePattern"}
   (apply base/call-function @ShapePattern-fnptr* args)))

(defonce ^:private TupleGetItemPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.TupleGetItemPattern")))
(defn TupleGetItemPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.TupleGetItemPattern"}
   (apply base/call-function @TupleGetItemPattern-fnptr* args)))

(defonce ^:private TuplePattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.TuplePattern")))
(defn TuplePattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.TuplePattern"}
   (apply base/call-function @TuplePattern-fnptr* args)))

(defonce ^:private TypePattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.TypePattern")))
(defn TypePattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.TypePattern"}
   (apply base/call-function @TypePattern-fnptr* args)))

(defonce ^:private VarPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.VarPattern")))
(defn VarPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.VarPattern"}
   (apply base/call-function @VarPattern-fnptr* args)))

(defonce ^:private WildcardPattern-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.WildcardPattern")))
(defn WildcardPattern
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.WildcardPattern"}
   (apply base/call-function @WildcardPattern-fnptr* args)))

(defonce ^:private match-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.match")))
(defn match
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.match"}
   (apply base/call-function @match-fnptr* args)))

(defonce ^:private partition-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.partition")))
(defn partition
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.partition"}
   (apply base/call-function @partition-fnptr* args)))

(defonce ^:private rewrite-fnptr* (delay (base/name->global-function "relay.dataflow_pattern.rewrite")))
(defn rewrite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.dataflow_pattern.rewrite"}
   (apply base/call-function @rewrite-fnptr* args)))

