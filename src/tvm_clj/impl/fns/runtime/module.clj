(ns tvm-clj.impl.fns.runtime.module
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private loadbinary_GraphRuntimeFactory-fnptr* (delay (base/name->global-function "runtime.module.loadbinary_GraphRuntimeFactory")))
(defn loadbinary_GraphRuntimeFactory
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.module.loadbinary_GraphRuntimeFactory"}
   (apply base/call-function @loadbinary_GraphRuntimeFactory-fnptr* args)))

(defonce ^:private loadbinary_metadata-fnptr* (delay (base/name->global-function "runtime.module.loadbinary_metadata")))
(defn loadbinary_metadata
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.module.loadbinary_metadata"}
   (apply base/call-function @loadbinary_metadata-fnptr* args)))

(defonce ^:private loadfile_ll-fnptr* (delay (base/name->global-function "runtime.module.loadfile_ll")))
(defn loadfile_ll
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.module.loadfile_ll"}
   (apply base/call-function @loadfile_ll-fnptr* args)))

(defonce ^:private loadfile_so-fnptr* (delay (base/name->global-function "runtime.module.loadfile_so")))
(defn loadfile_so
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.module.loadfile_so"}
   (apply base/call-function @loadfile_so-fnptr* args)))

(defonce ^:private loadfile_stackvm-fnptr* (delay (base/name->global-function "runtime.module.loadfile_stackvm")))
(defn loadfile_stackvm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "runtime.module.loadfile_stackvm"}
   (apply base/call-function @loadfile_stackvm-fnptr* args)))

