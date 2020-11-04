(ns tvm-clj.jna.fns.relay.backend
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.backend.CreateInterpreter"))]
  (defn CreateInterpreter
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend.CreateInterpreter"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend.GraphPlanMemory"))]
  (defn GraphPlanMemory
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend.GraphPlanMemory"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineClear"))]
  (defn _CompileEngineClear
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileEngineClear"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineGlobal"))]
  (defn _CompileEngineGlobal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileEngineGlobal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineJIT"))]
  (defn _CompileEngineJIT
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileEngineJIT"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineListItems"))]
  (defn _CompileEngineListItems
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileEngineListItems"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineLower"))]
  (defn _CompileEngineLower
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileEngineLower"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileEngineLowerShapeFunc"))]
  (defn _CompileEngineLowerShapeFunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileEngineLowerShapeFunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._CompileLowerExternalFunctions"))]
  (defn _CompileLowerExternalFunctions
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._CompileLowerExternalFunctions"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._make_CCacheKey"))]
  (defn _make_CCacheKey
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._make_CCacheKey"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.backend._make_LoweredOutput"))]
  (defn _make_LoweredOutput
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.backend._make_LoweredOutput"}
     (apply jna-base/call-function @gfn* args))))

