(ns tvm-clj.jna.fns.relay.qnn.op._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} add
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} concatenate
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.concatenate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} conv2d
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.conv2d"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dense
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} dequantize
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.dequantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} mul
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.mul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} quantize
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.quantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} requantize
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.requantize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} subtract
(let [gfn* (delay (jna-base/name->global-function "relay.qnn.op._make.subtract"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

