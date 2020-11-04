(ns tvm-clj.jna.fns.node
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Array
(let [gfn* (delay (jna-base/name->global-function "node.Array"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ArrayGetItem
(let [gfn* (delay (jna-base/name->global-function "node.ArrayGetItem"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ArraySize
(let [gfn* (delay (jna-base/name->global-function "node.ArraySize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AsRepr
(let [gfn* (delay (jna-base/name->global-function "node.AsRepr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LargeUIntImm
(let [gfn* (delay (jna-base/name->global-function "node.LargeUIntImm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LoadJSON
(let [gfn* (delay (jna-base/name->global-function "node.LoadJSON"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MakeNode
(let [gfn* (delay (jna-base/name->global-function "node.MakeNode"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Map
(let [gfn* (delay (jna-base/name->global-function "node.Map"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MapCount
(let [gfn* (delay (jna-base/name->global-function "node.MapCount"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MapGetItem
(let [gfn* (delay (jna-base/name->global-function "node.MapGetItem"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MapItems
(let [gfn* (delay (jna-base/name->global-function "node.MapItems"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} MapSize
(let [gfn* (delay (jna-base/name->global-function "node.MapSize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NodeGetAttr
(let [gfn* (delay (jna-base/name->global-function "node.NodeGetAttr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NodeListAttrNames
(let [gfn* (delay (jna-base/name->global-function "node.NodeListAttrNames"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SaveJSON
(let [gfn* (delay (jna-base/name->global-function "node.SaveJSON"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StructuralEqual
(let [gfn* (delay (jna-base/name->global-function "node.StructuralEqual"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StructuralHash
(let [gfn* (delay (jna-base/name->global-function "node.StructuralHash"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _const
(let [gfn* (delay (jna-base/name->global-function "node._const"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

