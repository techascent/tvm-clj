(ns tvm-clj.jna.fns.runtime.module
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} loadbinary_GraphRuntimeFactory
(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadbinary_GraphRuntimeFactory"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} loadbinary_metadata
(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadbinary_metadata"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} loadfile_ll
(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadfile_ll"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} loadfile_so
(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadfile_so"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} loadfile_stackvm
(let [gfn* (delay (jna-base/name->global-function "runtime.module.loadfile_stackvm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

