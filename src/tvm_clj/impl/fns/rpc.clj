(ns tvm-clj.jna.fns.rpc
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "rpc.Connect"))]
  (defn Connect
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.Connect"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.CreateEventDrivenServer"))]
  (defn CreateEventDrivenServer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.CreateEventDrivenServer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.CreatePipeClient"))]
  (defn CreatePipeClient
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.CreatePipeClient"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.ImportRemoteModule"))]
  (defn ImportRemoteModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.ImportRemoteModule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.LoadRemoteModule"))]
  (defn LoadRemoteModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.LoadRemoteModule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.LocalSession"))]
  (defn LocalSession
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.LocalSession"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.ServerLoop"))]
  (defn ServerLoop
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.ServerLoop"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "rpc.SessTableIndex"))]
  (defn SessTableIndex
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "rpc.SessTableIndex"}
     (apply jna-base/call-function @gfn* args))))

