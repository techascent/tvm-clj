(ns tvm-clj.jna.fns.relay.dataflow_pattern
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AltPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.AltPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AttrPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.AttrPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CallPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.CallPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ConstantPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.ConstantPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DFPatternCallback
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.DFPatternCallback"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DataTypePattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.DataTypePattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} DominatorPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.DominatorPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ExprPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.ExprPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ShapePattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.ShapePattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TupleGetItemPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.TupleGetItemPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TuplePattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.TuplePattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TypePattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.TypePattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} VarPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.VarPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} WildcardPattern
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.WildcardPattern"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} match
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.match"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} partition
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.partition"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} rewrite
(let [gfn* (delay (jna-base/name->global-function "relay.dataflow_pattern.rewrite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

