(ns tvm-clj.jna.fns.relay._quantize
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.CreateStatsCollector"))]
  (defn CreateStatsCollector
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.CreateStatsCollector"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.FindScaleByKLMinimization"))]
  (defn FindScaleByKLMinimization
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.FindScaleByKLMinimization"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.QuantizeAnnotate"))]
  (defn QuantizeAnnotate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.QuantizeAnnotate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.QuantizePartition"))]
  (defn QuantizePartition
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.QuantizePartition"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.QuantizeRealize"))]
  (defn QuantizeRealize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.QuantizeRealize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize._EnterQConfigScope"))]
  (defn _EnterQConfigScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize._EnterQConfigScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize._ExitQConfigScope"))]
  (defn _ExitQConfigScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize._ExitQConfigScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize._GetCurrentQConfig"))]
  (defn _GetCurrentQConfig
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize._GetCurrentQConfig"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.make_annotate_expr"))]
  (defn make_annotate_expr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.make_annotate_expr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.make_partition_expr"))]
  (defn make_partition_expr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.make_partition_expr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay._quantize.simulated_quantize"))]
  (defn simulated_quantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay._quantize.simulated_quantize"}
     (apply jna-base/call-function @gfn* args))))

