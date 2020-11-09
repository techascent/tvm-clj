(ns tvm-clj.jna.fns.tvm.rpc.server
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.ImportModule"))]
  (defn ImportModule
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.rpc.server.ImportModule"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.ModuleGetFunction"))]
  (defn ModuleGetFunction
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.rpc.server.ModuleGetFunction"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.download"))]
  (defn download
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.rpc.server.download"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.remove"))]
  (defn remove
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.rpc.server.remove"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.rpc.server.upload"))]
  (defn upload
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.rpc.server.upload"}
     (apply jna-base/call-function @gfn* args))))

