(ns tvm-clj.jna.fns.target
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "target.Build"))]
  (defn Build
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.Build"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncCallFunc"))]
  (defn GenericFuncCallFunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.GenericFuncCallFunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncCreate"))]
  (defn GenericFuncCreate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.GenericFuncCreate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncGetGlobal"))]
  (defn GenericFuncGetGlobal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.GenericFuncGetGlobal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncRegisterFunc"))]
  (defn GenericFuncRegisterFunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.GenericFuncRegisterFunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.GenericFuncSetDefault"))]
  (defn GenericFuncSetDefault
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.GenericFuncSetDefault"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.Target"))]
  (defn Target
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.Target"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.TargetCurrent"))]
  (defn TargetCurrent
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.TargetCurrent"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.TargetEnterScope"))]
  (defn TargetEnterScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.TargetEnterScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.TargetExitScope"))]
  (defn TargetExitScope
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.TargetExitScope"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.TargetExport"))]
  (defn TargetExport
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.TargetExport"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.TargetTagAddTag"))]
  (defn TargetTagAddTag
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.TargetTagAddTag"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.TargetTagListTags"))]
  (defn TargetTagListTags
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.TargetTagListTags"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.llvm_lookup_intrinsic_id"))]
  (defn llvm_lookup_intrinsic_id
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.llvm_lookup_intrinsic_id"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "target.llvm_version_major"))]
  (defn llvm_version_major
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "target.llvm_version_major"}
     (apply jna-base/call-function @gfn* args))))

