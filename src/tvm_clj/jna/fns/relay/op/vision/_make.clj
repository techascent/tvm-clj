(ns tvm-clj.jna.fns.relay.op.vision._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} get_valid_counts
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.get_valid_counts"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} multibox_prior
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.multibox_prior"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} multibox_transform_loc
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.multibox_transform_loc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} non_max_suppression
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.non_max_suppression"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} proposal
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.proposal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} roi_align
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.roi_align"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} roi_pool
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.roi_pool"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} yolo_reorg
(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.yolo_reorg"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

