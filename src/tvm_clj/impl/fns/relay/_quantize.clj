(ns tvm-clj.impl.fns.relay._quantize
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private CreateStatsCollector-fnptr* (delay (base/name->global-function "relay._quantize.CreateStatsCollector")))
(defn CreateStatsCollector
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.CreateStatsCollector"}
   (apply base/call-function @CreateStatsCollector-fnptr* args)))

(defonce ^:private FindScaleByKLMinimization-fnptr* (delay (base/name->global-function "relay._quantize.FindScaleByKLMinimization")))
(defn FindScaleByKLMinimization
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.FindScaleByKLMinimization"}
   (apply base/call-function @FindScaleByKLMinimization-fnptr* args)))

(defonce ^:private QuantizeAnnotate-fnptr* (delay (base/name->global-function "relay._quantize.QuantizeAnnotate")))
(defn QuantizeAnnotate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.QuantizeAnnotate"}
   (apply base/call-function @QuantizeAnnotate-fnptr* args)))

(defonce ^:private QuantizePartition-fnptr* (delay (base/name->global-function "relay._quantize.QuantizePartition")))
(defn QuantizePartition
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.QuantizePartition"}
   (apply base/call-function @QuantizePartition-fnptr* args)))

(defonce ^:private QuantizeRealize-fnptr* (delay (base/name->global-function "relay._quantize.QuantizeRealize")))
(defn QuantizeRealize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.QuantizeRealize"}
   (apply base/call-function @QuantizeRealize-fnptr* args)))

(defonce ^:private _EnterQConfigScope-fnptr* (delay (base/name->global-function "relay._quantize._EnterQConfigScope")))
(defn _EnterQConfigScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize._EnterQConfigScope"}
   (apply base/call-function @_EnterQConfigScope-fnptr* args)))

(defonce ^:private _ExitQConfigScope-fnptr* (delay (base/name->global-function "relay._quantize._ExitQConfigScope")))
(defn _ExitQConfigScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize._ExitQConfigScope"}
   (apply base/call-function @_ExitQConfigScope-fnptr* args)))

(defonce ^:private _GetCurrentQConfig-fnptr* (delay (base/name->global-function "relay._quantize._GetCurrentQConfig")))
(defn _GetCurrentQConfig
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize._GetCurrentQConfig"}
   (apply base/call-function @_GetCurrentQConfig-fnptr* args)))

(defonce ^:private make_annotate_expr-fnptr* (delay (base/name->global-function "relay._quantize.make_annotate_expr")))
(defn make_annotate_expr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.make_annotate_expr"}
   (apply base/call-function @make_annotate_expr-fnptr* args)))

(defonce ^:private make_partition_expr-fnptr* (delay (base/name->global-function "relay._quantize.make_partition_expr")))
(defn make_partition_expr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.make_partition_expr"}
   (apply base/call-function @make_partition_expr-fnptr* args)))

(defonce ^:private simulated_quantize-fnptr* (delay (base/name->global-function "relay._quantize.simulated_quantize")))
(defn simulated_quantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay._quantize.simulated_quantize"}
   (apply base/call-function @simulated_quantize-fnptr* args)))

