(ns tvm-clj.jna.fns.target
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Build
(let [gfn* (delay (jna-base/name->global-function "target.Build"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GenericFuncCallFunc
(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncCallFunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GenericFuncCreate
(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncCreate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GenericFuncGetGlobal
(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncGetGlobal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GenericFuncRegisterFunc
(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncRegisterFunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GenericFuncSetDefault
(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncSetDefault"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Target
(let [gfn* (delay (jna-base/name->global-function "target.Target"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TargetCurrent
(let [gfn* (delay (jna-base/name->global-function "target.TargetCurrent"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TargetEnterScope
(let [gfn* (delay (jna-base/name->global-function "target.TargetEnterScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TargetExitScope
(let [gfn* (delay (jna-base/name->global-function "target.TargetExitScope"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TargetExport
(let [gfn* (delay (jna-base/name->global-function "target.TargetExport"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TargetTagAddTag
(let [gfn* (delay (jna-base/name->global-function "target.TargetTagAddTag"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TargetTagListTags
(let [gfn* (delay (jna-base/name->global-function "target.TargetTagListTags"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} llvm_lookup_intrinsic_id
(let [gfn* (delay (jna-base/name->global-function "target.llvm_lookup_intrinsic_id"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} llvm_version_major
(let [gfn* (delay (jna-base/name->global-function "target.llvm_version_major"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

