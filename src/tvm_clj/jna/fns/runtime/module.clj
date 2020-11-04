(ns tvm-clj.jna.fns.runtime.module
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadbinary_GraphRuntimeFactory"))]
  (defn loadbinary_GraphRuntimeFactory
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.module.loadbinary_GraphRuntimeFactory"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadbinary_metadata"))]
  (defn loadbinary_metadata
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.module.loadbinary_metadata"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadfile_ll"))]
  (defn loadfile_ll
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.module.loadfile_ll"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadfile_so"))]
  (defn loadfile_so
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.module.loadfile_so"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadfile_stackvm"))]
  (defn loadfile_stackvm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "runtime.module.loadfile_stackvm"}
     (apply jna-base/call-function @gfn* args))))

