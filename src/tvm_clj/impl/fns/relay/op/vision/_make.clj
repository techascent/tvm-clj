(ns tvm-clj.jna.fns.relay.op.vision._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.get_valid_counts"))]
  (defn get_valid_counts
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.get_valid_counts"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.multibox_prior"))]
  (defn multibox_prior
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.multibox_prior"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.multibox_transform_loc"))]
  (defn multibox_transform_loc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.multibox_transform_loc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.non_max_suppression"))]
  (defn non_max_suppression
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.non_max_suppression"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.proposal"))]
  (defn proposal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.proposal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.roi_align"))]
  (defn roi_align
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.roi_align"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.roi_pool"))]
  (defn roi_pool
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.roi_pool"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op.vision._make.yolo_reorg"))]
  (defn yolo_reorg
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op.vision._make.yolo_reorg"}
     (apply jna-base/call-function @gfn* args))))

