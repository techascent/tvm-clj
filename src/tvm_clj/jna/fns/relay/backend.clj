(ns tvm-clj.jna.fns.relay.backend
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateInterpreter
(let [gfn* (delay (jna-base/name->global-function "relay.backend.CreateInterpreter"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GraphPlanMemory
(let [gfn* (delay (jna-base/name->global-function "relay.backend.GraphPlanMemory"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileEngineClear
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineClear"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileEngineGlobal
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineGlobal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileEngineJIT
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineJIT"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileEngineListItems
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineListItems"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileEngineLower
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineLower"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileEngineLowerShapeFunc
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineLowerShapeFunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _CompileLowerExternalFunctions
(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileLowerExternalFunctions"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _make_CCacheKey
(let [gfn* (delay (jna-base/name->global-function "relay.backend._make_CCacheKey"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _make_LoweredOutput
(let [gfn* (delay (jna-base/name->global-function "relay.backend._make_LoweredOutput"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

