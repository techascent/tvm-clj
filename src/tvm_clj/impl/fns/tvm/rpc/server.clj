(ns tvm-clj.impl.fns.tvm.rpc.server
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ImportModule-fnptr* (delay (base/name->global-function "tvm.rpc.server.ImportModule")))
(defn ImportModule
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.rpc.server.ImportModule"}
   (apply base/call-function @ImportModule-fnptr* args)))

(defonce ^:private ModuleGetFunction-fnptr* (delay (base/name->global-function "tvm.rpc.server.ModuleGetFunction")))
(defn ModuleGetFunction
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.rpc.server.ModuleGetFunction"}
   (apply base/call-function @ModuleGetFunction-fnptr* args)))

(defonce ^:private download-fnptr* (delay (base/name->global-function "tvm.rpc.server.download")))
(defn download
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.rpc.server.download"}
   (apply base/call-function @download-fnptr* args)))

(defonce ^:private remove-fnptr* (delay (base/name->global-function "tvm.rpc.server.remove")))
(defn remove
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.rpc.server.remove"}
   (apply base/call-function @remove-fnptr* args)))

(defonce ^:private upload-fnptr* (delay (base/name->global-function "tvm.rpc.server.upload")))
(defn upload
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.rpc.server.upload"}
   (apply base/call-function @upload-fnptr* args)))

