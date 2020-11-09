(ns tvm-clj.impl.fns.relay.backend
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private CreateInterpreter-fnptr* (delay (base/name->global-function "relay.backend.CreateInterpreter")))
(defn CreateInterpreter
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend.CreateInterpreter"}
   (apply base/call-function @CreateInterpreter-fnptr* args)))

(defonce ^:private GraphPlanMemory-fnptr* (delay (base/name->global-function "relay.backend.GraphPlanMemory")))
(defn GraphPlanMemory
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend.GraphPlanMemory"}
   (apply base/call-function @GraphPlanMemory-fnptr* args)))

(defonce ^:private _CompileEngineClear-fnptr* (delay (base/name->global-function "relay.backend._CompileEngineClear")))
(defn _CompileEngineClear
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileEngineClear"}
   (apply base/call-function @_CompileEngineClear-fnptr* args)))

(defonce ^:private _CompileEngineGlobal-fnptr* (delay (base/name->global-function "relay.backend._CompileEngineGlobal")))
(defn _CompileEngineGlobal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileEngineGlobal"}
   (apply base/call-function @_CompileEngineGlobal-fnptr* args)))

(defonce ^:private _CompileEngineJIT-fnptr* (delay (base/name->global-function "relay.backend._CompileEngineJIT")))
(defn _CompileEngineJIT
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileEngineJIT"}
   (apply base/call-function @_CompileEngineJIT-fnptr* args)))

(defonce ^:private _CompileEngineListItems-fnptr* (delay (base/name->global-function "relay.backend._CompileEngineListItems")))
(defn _CompileEngineListItems
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileEngineListItems"}
   (apply base/call-function @_CompileEngineListItems-fnptr* args)))

(defonce ^:private _CompileEngineLower-fnptr* (delay (base/name->global-function "relay.backend._CompileEngineLower")))
(defn _CompileEngineLower
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileEngineLower"}
   (apply base/call-function @_CompileEngineLower-fnptr* args)))

(defonce ^:private _CompileEngineLowerShapeFunc-fnptr* (delay (base/name->global-function "relay.backend._CompileEngineLowerShapeFunc")))
(defn _CompileEngineLowerShapeFunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileEngineLowerShapeFunc"}
   (apply base/call-function @_CompileEngineLowerShapeFunc-fnptr* args)))

(defonce ^:private _CompileLowerExternalFunctions-fnptr* (delay (base/name->global-function "relay.backend._CompileLowerExternalFunctions")))
(defn _CompileLowerExternalFunctions
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._CompileLowerExternalFunctions"}
   (apply base/call-function @_CompileLowerExternalFunctions-fnptr* args)))

(defonce ^:private _make_CCacheKey-fnptr* (delay (base/name->global-function "relay.backend._make_CCacheKey")))
(defn _make_CCacheKey
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._make_CCacheKey"}
   (apply base/call-function @_make_CCacheKey-fnptr* args)))

(defonce ^:private _make_LoweredOutput-fnptr* (delay (base/name->global-function "relay.backend._make_LoweredOutput")))
(defn _make_LoweredOutput
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.backend._make_LoweredOutput"}
   (apply base/call-function @_make_LoweredOutput-fnptr* args)))

