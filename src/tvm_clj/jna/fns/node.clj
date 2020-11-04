(ns tvm-clj.jna.fns.node
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "node.Array"))]
  (defn Array
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.Array"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.ArrayGetItem"))]
  (defn ArrayGetItem
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.ArrayGetItem"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.ArraySize"))]
  (defn ArraySize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.ArraySize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.AsRepr"))]
  (defn AsRepr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.AsRepr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.LargeUIntImm"))]
  (defn LargeUIntImm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.LargeUIntImm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.LoadJSON"))]
  (defn LoadJSON
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.LoadJSON"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.MakeNode"))]
  (defn MakeNode
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.MakeNode"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.Map"))]
  (defn Map
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.Map"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.MapCount"))]
  (defn MapCount
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.MapCount"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.MapGetItem"))]
  (defn MapGetItem
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.MapGetItem"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.MapItems"))]
  (defn MapItems
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.MapItems"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.MapSize"))]
  (defn MapSize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.MapSize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.NodeGetAttr"))]
  (defn NodeGetAttr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.NodeGetAttr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.NodeListAttrNames"))]
  (defn NodeListAttrNames
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.NodeListAttrNames"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.SaveJSON"))]
  (defn SaveJSON
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.SaveJSON"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.StructuralEqual"))]
  (defn StructuralEqual
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.StructuralEqual"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node.StructuralHash"))]
  (defn StructuralHash
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node.StructuralHash"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "node._const"))]
  (defn _const
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "node._const"}
     (apply jna-base/call-function @gfn* args))))

