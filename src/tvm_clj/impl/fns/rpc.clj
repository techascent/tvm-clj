(ns tvm-clj.impl.fns.rpc
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private Connect-fnptr* (delay (base/name->global-function "rpc.Connect")))
(defn Connect
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.Connect"}
   (apply base/call-function @Connect-fnptr* args)))

(defonce ^:private CreateEventDrivenServer-fnptr* (delay (base/name->global-function "rpc.CreateEventDrivenServer")))
(defn CreateEventDrivenServer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.CreateEventDrivenServer"}
   (apply base/call-function @CreateEventDrivenServer-fnptr* args)))

(defonce ^:private CreatePipeClient-fnptr* (delay (base/name->global-function "rpc.CreatePipeClient")))
(defn CreatePipeClient
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.CreatePipeClient"}
   (apply base/call-function @CreatePipeClient-fnptr* args)))

(defonce ^:private ImportRemoteModule-fnptr* (delay (base/name->global-function "rpc.ImportRemoteModule")))
(defn ImportRemoteModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.ImportRemoteModule"}
   (apply base/call-function @ImportRemoteModule-fnptr* args)))

(defonce ^:private LoadRemoteModule-fnptr* (delay (base/name->global-function "rpc.LoadRemoteModule")))
(defn LoadRemoteModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.LoadRemoteModule"}
   (apply base/call-function @LoadRemoteModule-fnptr* args)))

(defonce ^:private LocalSession-fnptr* (delay (base/name->global-function "rpc.LocalSession")))
(defn LocalSession
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.LocalSession"}
   (apply base/call-function @LocalSession-fnptr* args)))

(defonce ^:private ServerLoop-fnptr* (delay (base/name->global-function "rpc.ServerLoop")))
(defn ServerLoop
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.ServerLoop"}
   (apply base/call-function @ServerLoop-fnptr* args)))

(defonce ^:private SessTableIndex-fnptr* (delay (base/name->global-function "rpc.SessTableIndex")))
(defn SessTableIndex
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "rpc.SessTableIndex"}
   (apply base/call-function @SessTableIndex-fnptr* args)))

