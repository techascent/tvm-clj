(ns tvm-clj.impl.fns.node
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private Array-fnptr* (delay (base/name->global-function "node.Array")))
(defn Array
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.Array"}
   (apply base/call-function @Array-fnptr* args)))

(defonce ^:private ArrayGetItem-fnptr* (delay (base/name->global-function "node.ArrayGetItem")))
(defn ArrayGetItem
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.ArrayGetItem"}
   (apply base/call-function @ArrayGetItem-fnptr* args)))

(defonce ^:private ArraySize-fnptr* (delay (base/name->global-function "node.ArraySize")))
(defn ArraySize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.ArraySize"}
   (apply base/call-function @ArraySize-fnptr* args)))

(defonce ^:private AsRepr-fnptr* (delay (base/name->global-function "node.AsRepr")))
(defn AsRepr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.AsRepr"}
   (apply base/call-function @AsRepr-fnptr* args)))

(defonce ^:private LargeUIntImm-fnptr* (delay (base/name->global-function "node.LargeUIntImm")))
(defn LargeUIntImm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.LargeUIntImm"}
   (apply base/call-function @LargeUIntImm-fnptr* args)))

(defonce ^:private LoadJSON-fnptr* (delay (base/name->global-function "node.LoadJSON")))
(defn LoadJSON
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.LoadJSON"}
   (apply base/call-function @LoadJSON-fnptr* args)))

(defonce ^:private MakeNode-fnptr* (delay (base/name->global-function "node.MakeNode")))
(defn MakeNode
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.MakeNode"}
   (apply base/call-function @MakeNode-fnptr* args)))

(defonce ^:private Map-fnptr* (delay (base/name->global-function "node.Map")))
(defn Map
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.Map"}
   (apply base/call-function @Map-fnptr* args)))

(defonce ^:private MapCount-fnptr* (delay (base/name->global-function "node.MapCount")))
(defn MapCount
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.MapCount"}
   (apply base/call-function @MapCount-fnptr* args)))

(defonce ^:private MapGetItem-fnptr* (delay (base/name->global-function "node.MapGetItem")))
(defn MapGetItem
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.MapGetItem"}
   (apply base/call-function @MapGetItem-fnptr* args)))

(defonce ^:private MapItems-fnptr* (delay (base/name->global-function "node.MapItems")))
(defn MapItems
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.MapItems"}
   (apply base/call-function @MapItems-fnptr* args)))

(defonce ^:private MapSize-fnptr* (delay (base/name->global-function "node.MapSize")))
(defn MapSize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.MapSize"}
   (apply base/call-function @MapSize-fnptr* args)))

(defonce ^:private NodeGetAttr-fnptr* (delay (base/name->global-function "node.NodeGetAttr")))
(defn NodeGetAttr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.NodeGetAttr"}
   (apply base/call-function @NodeGetAttr-fnptr* args)))

(defonce ^:private NodeListAttrNames-fnptr* (delay (base/name->global-function "node.NodeListAttrNames")))
(defn NodeListAttrNames
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.NodeListAttrNames"}
   (apply base/call-function @NodeListAttrNames-fnptr* args)))

(defonce ^:private SaveJSON-fnptr* (delay (base/name->global-function "node.SaveJSON")))
(defn SaveJSON
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.SaveJSON"}
   (apply base/call-function @SaveJSON-fnptr* args)))

(defonce ^:private StructuralEqual-fnptr* (delay (base/name->global-function "node.StructuralEqual")))
(defn StructuralEqual
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.StructuralEqual"}
   (apply base/call-function @StructuralEqual-fnptr* args)))

(defonce ^:private StructuralHash-fnptr* (delay (base/name->global-function "node.StructuralHash")))
(defn StructuralHash
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node.StructuralHash"}
   (apply base/call-function @StructuralHash-fnptr* args)))

(defonce ^:private _const-fnptr* (delay (base/name->global-function "node._const")))
(defn _const
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "node._const"}
   (apply base/call-function @_const-fnptr* args)))

