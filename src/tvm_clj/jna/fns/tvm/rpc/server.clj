(ns tvm-clj.jna.fns.tvm.rpc.server
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ImportModule
(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.ImportModule"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ModuleGetFunction
(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.ModuleGetFunction"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} download
(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.download"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} remove
(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.remove"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} upload
(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.upload"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

