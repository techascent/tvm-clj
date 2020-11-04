(ns tvm-clj.jna.fns.relay._quantize
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateStatsCollector
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.CreateStatsCollector"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FindScaleByKLMinimization
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.FindScaleByKLMinimization"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QuantizeAnnotate
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.QuantizeAnnotate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QuantizePartition
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.QuantizePartition"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} QuantizeRealize
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.QuantizeRealize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _EnterQConfigScope
(let [gfn* (delay (jna-base/name->global-function "relay._quantize._EnterQConfigScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _ExitQConfigScope
(let [gfn* (delay (jna-base/name->global-function "relay._quantize._ExitQConfigScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _GetCurrentQConfig
(let [gfn* (delay (jna-base/name->global-function "relay._quantize._GetCurrentQConfig"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} make_annotate_expr
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.make_annotate_expr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} make_partition_expr
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.make_partition_expr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} simulated_quantize
(let [gfn* (delay (jna-base/name->global-function "relay._quantize.simulated_quantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

