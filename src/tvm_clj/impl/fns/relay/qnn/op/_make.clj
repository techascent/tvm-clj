(ns tvm-clj.impl.fns.relay.qnn.op._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private add-fnptr* (delay (base/name->global-function "relay.qnn.op._make.add")))
(defn add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.add"}
   (apply base/call-function @add-fnptr* args)))

(defonce ^:private concatenate-fnptr* (delay (base/name->global-function "relay.qnn.op._make.concatenate")))
(defn concatenate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.concatenate"}
   (apply base/call-function @concatenate-fnptr* args)))

(defonce ^:private conv2d-fnptr* (delay (base/name->global-function "relay.qnn.op._make.conv2d")))
(defn conv2d
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.conv2d"}
   (apply base/call-function @conv2d-fnptr* args)))

(defonce ^:private dense-fnptr* (delay (base/name->global-function "relay.qnn.op._make.dense")))
(defn dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.dense"}
   (apply base/call-function @dense-fnptr* args)))

(defonce ^:private dequantize-fnptr* (delay (base/name->global-function "relay.qnn.op._make.dequantize")))
(defn dequantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.dequantize"}
   (apply base/call-function @dequantize-fnptr* args)))

(defonce ^:private mul-fnptr* (delay (base/name->global-function "relay.qnn.op._make.mul")))
(defn mul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.mul"}
   (apply base/call-function @mul-fnptr* args)))

(defonce ^:private quantize-fnptr* (delay (base/name->global-function "relay.qnn.op._make.quantize")))
(defn quantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.quantize"}
   (apply base/call-function @quantize-fnptr* args)))

(defonce ^:private requantize-fnptr* (delay (base/name->global-function "relay.qnn.op._make.requantize")))
(defn requantize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.requantize"}
   (apply base/call-function @requantize-fnptr* args)))

(defonce ^:private subtract-fnptr* (delay (base/name->global-function "relay.qnn.op._make.subtract")))
(defn subtract
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.qnn.op._make.subtract"}
   (apply base/call-function @subtract-fnptr* args)))

