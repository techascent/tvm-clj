(ns tvm-clj.impl.fns.target
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private Build-fnptr* (delay (base/name->global-function "target.Build")))
(defn Build
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.Build"}
   (apply base/call-function @Build-fnptr* args)))

(defonce ^:private GenericFuncCallFunc-fnptr* (delay (base/name->global-function "target.GenericFuncCallFunc")))
(defn GenericFuncCallFunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.GenericFuncCallFunc"}
   (apply base/call-function @GenericFuncCallFunc-fnptr* args)))

(defonce ^:private GenericFuncCreate-fnptr* (delay (base/name->global-function "target.GenericFuncCreate")))
(defn GenericFuncCreate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.GenericFuncCreate"}
   (apply base/call-function @GenericFuncCreate-fnptr* args)))

(defonce ^:private GenericFuncGetGlobal-fnptr* (delay (base/name->global-function "target.GenericFuncGetGlobal")))
(defn GenericFuncGetGlobal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.GenericFuncGetGlobal"}
   (apply base/call-function @GenericFuncGetGlobal-fnptr* args)))

(defonce ^:private GenericFuncRegisterFunc-fnptr* (delay (base/name->global-function "target.GenericFuncRegisterFunc")))
(defn GenericFuncRegisterFunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.GenericFuncRegisterFunc"}
   (apply base/call-function @GenericFuncRegisterFunc-fnptr* args)))

(defonce ^:private GenericFuncSetDefault-fnptr* (delay (base/name->global-function "target.GenericFuncSetDefault")))
(defn GenericFuncSetDefault
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.GenericFuncSetDefault"}
   (apply base/call-function @GenericFuncSetDefault-fnptr* args)))

(defonce ^:private Target-fnptr* (delay (base/name->global-function "target.Target")))
(defn Target
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.Target"}
   (apply base/call-function @Target-fnptr* args)))

(defonce ^:private TargetCurrent-fnptr* (delay (base/name->global-function "target.TargetCurrent")))
(defn TargetCurrent
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.TargetCurrent"}
   (apply base/call-function @TargetCurrent-fnptr* args)))

(defonce ^:private TargetEnterScope-fnptr* (delay (base/name->global-function "target.TargetEnterScope")))
(defn TargetEnterScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.TargetEnterScope"}
   (apply base/call-function @TargetEnterScope-fnptr* args)))

(defonce ^:private TargetExitScope-fnptr* (delay (base/name->global-function "target.TargetExitScope")))
(defn TargetExitScope
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.TargetExitScope"}
   (apply base/call-function @TargetExitScope-fnptr* args)))

(defonce ^:private TargetExport-fnptr* (delay (base/name->global-function "target.TargetExport")))
(defn TargetExport
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.TargetExport"}
   (apply base/call-function @TargetExport-fnptr* args)))

(defonce ^:private TargetTagAddTag-fnptr* (delay (base/name->global-function "target.TargetTagAddTag")))
(defn TargetTagAddTag
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.TargetTagAddTag"}
   (apply base/call-function @TargetTagAddTag-fnptr* args)))

(defonce ^:private TargetTagListTags-fnptr* (delay (base/name->global-function "target.TargetTagListTags")))
(defn TargetTagListTags
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.TargetTagListTags"}
   (apply base/call-function @TargetTagListTags-fnptr* args)))

(defonce ^:private llvm_lookup_intrinsic_id-fnptr* (delay (base/name->global-function "target.llvm_lookup_intrinsic_id")))
(defn llvm_lookup_intrinsic_id
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.llvm_lookup_intrinsic_id"}
   (apply base/call-function @llvm_lookup_intrinsic_id-fnptr* args)))

(defonce ^:private llvm_version_major-fnptr* (delay (base/name->global-function "target.llvm_version_major")))
(defn llvm_version_major
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "target.llvm_version_major"}
   (apply base/call-function @llvm_version_major-fnptr* args)))

