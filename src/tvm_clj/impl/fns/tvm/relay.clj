(ns tvm-clj.impl.fns.tvm.relay
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private _load_param_dict-fnptr* (delay (base/name->global-function "tvm.relay._load_param_dict")))
(defn _load_param_dict
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay._load_param_dict"}
   (apply base/call-function @_load_param_dict-fnptr* args)))

(defonce ^:private _save_param_dict-fnptr* (delay (base/name->global-function "tvm.relay._save_param_dict")))
(defn _save_param_dict
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tvm.relay._save_param_dict"}
   (apply base/call-function @_save_param_dict-fnptr* args)))

