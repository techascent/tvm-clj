(ns tvm-clj.jna.fns.relay.dataflow_pattern
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.AltPattern"))]
  (defn AltPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.AltPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.AttrPattern"))]
  (defn AttrPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.AttrPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.CallPattern"))]
  (defn CallPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.CallPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.ConstantPattern"))]
  (defn ConstantPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.ConstantPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.DFPatternCallback"))]
  (defn DFPatternCallback
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.DFPatternCallback"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.DataTypePattern"))]
  (defn DataTypePattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.DataTypePattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.DominatorPattern"))]
  (defn DominatorPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.DominatorPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.ExprPattern"))]
  (defn ExprPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.ExprPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.ShapePattern"))]
  (defn ShapePattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.ShapePattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.TupleGetItemPattern"))]
  (defn TupleGetItemPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.TupleGetItemPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.TuplePattern"))]
  (defn TuplePattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.TuplePattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.TypePattern"))]
  (defn TypePattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.TypePattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.VarPattern"))]
  (defn VarPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.VarPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.WildcardPattern"))]
  (defn WildcardPattern
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.WildcardPattern"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.match"))]
  (defn match
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.match"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.partition"))]
  (defn partition
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.partition"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.rewrite"))]
  (defn rewrite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.dataflow_pattern.rewrite"}
     (apply jna-base/call-function @gfn* args))))

