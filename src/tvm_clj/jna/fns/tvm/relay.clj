(ns tvm-clj.jna.fns.tvm.relay
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay._load_param_dict"))]
  (defn _load_param_dict
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay._load_param_dict"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tvm.relay._save_param_dict"))]
  (defn _save_param_dict
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tvm.relay._save_param_dict"}
     (apply jna-base/call-function @gfn* args))))

