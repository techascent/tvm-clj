(ns tvm-clj.jna.fns.relay.op.memory._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FlattenTupleType
(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.FlattenTupleType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FromTupleType
(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.FromTupleType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ToTupleType
(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.ToTupleType"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} alloc_storage
(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.alloc_storage"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} alloc_tensor
(let [gfn* (delay (jna-base/name->global-function "relay.op.memory._make.alloc_tensor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

