(ns tvm-clj.impl.fns.relay.op.vision._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private get_valid_counts-fnptr* (delay (base/name->global-function "relay.op.vision._make.get_valid_counts")))
(defn get_valid_counts
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.get_valid_counts"}
   (apply base/call-function @get_valid_counts-fnptr* args)))

(defonce ^:private multibox_prior-fnptr* (delay (base/name->global-function "relay.op.vision._make.multibox_prior")))
(defn multibox_prior
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.multibox_prior"}
   (apply base/call-function @multibox_prior-fnptr* args)))

(defonce ^:private multibox_transform_loc-fnptr* (delay (base/name->global-function "relay.op.vision._make.multibox_transform_loc")))
(defn multibox_transform_loc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.multibox_transform_loc"}
   (apply base/call-function @multibox_transform_loc-fnptr* args)))

(defonce ^:private non_max_suppression-fnptr* (delay (base/name->global-function "relay.op.vision._make.non_max_suppression")))
(defn non_max_suppression
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.non_max_suppression"}
   (apply base/call-function @non_max_suppression-fnptr* args)))

(defonce ^:private proposal-fnptr* (delay (base/name->global-function "relay.op.vision._make.proposal")))
(defn proposal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.proposal"}
   (apply base/call-function @proposal-fnptr* args)))

(defonce ^:private roi_align-fnptr* (delay (base/name->global-function "relay.op.vision._make.roi_align")))
(defn roi_align
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.roi_align"}
   (apply base/call-function @roi_align-fnptr* args)))

(defonce ^:private roi_pool-fnptr* (delay (base/name->global-function "relay.op.vision._make.roi_pool")))
(defn roi_pool
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.roi_pool"}
   (apply base/call-function @roi_pool-fnptr* args)))

(defonce ^:private yolo_reorg-fnptr* (delay (base/name->global-function "relay.op.vision._make.yolo_reorg")))
(defn yolo_reorg
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op.vision._make.yolo_reorg"}
   (apply base/call-function @yolo_reorg-fnptr* args)))

