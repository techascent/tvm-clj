(ns tvm-clj.jna.fns.rpc
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Connect
(let [gfn* (delay (jna-base/name->global-function "rpc.Connect"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreateEventDrivenServer
(let [gfn* (delay (jna-base/name->global-function "rpc.CreateEventDrivenServer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CreatePipeClient
(let [gfn* (delay (jna-base/name->global-function "rpc.CreatePipeClient"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ImportRemoteModule
(let [gfn* (delay (jna-base/name->global-function "rpc.ImportRemoteModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LoadRemoteModule
(let [gfn* (delay (jna-base/name->global-function "rpc.LoadRemoteModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LocalSession
(let [gfn* (delay (jna-base/name->global-function "rpc.LocalSession"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ServerLoop
(let [gfn* (delay (jna-base/name->global-function "rpc.ServerLoop"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SessTableIndex
(let [gfn* (delay (jna-base/name->global-function "rpc.SessTableIndex"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

