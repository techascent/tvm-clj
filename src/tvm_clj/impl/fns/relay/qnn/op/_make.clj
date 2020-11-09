(ns tvm-clj.jna.fns.relay.qnn.op._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.add"))]
  (defn add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.concatenate"))]
  (defn concatenate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.concatenate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.conv2d"))]
  (defn conv2d
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.conv2d"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.dense"))]
  (defn dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.dequantize"))]
  (defn dequantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.dequantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.mul"))]
  (defn mul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.mul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.quantize"))]
  (defn quantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.quantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.requantize"))]
  (defn requantize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.requantize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.subtract"))]
  (defn subtract
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.qnn.op._make.subtract"}
     (apply jna-base/call-function @gfn* args))))

