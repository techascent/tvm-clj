(ns tvm-clj.jna.fns.tvm.relay
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _load_param_dict
(let [gfn* (delay (jna-base/name->global-function "tvm.relay._load_param_dict"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _save_param_dict
(let [gfn* (delay (jna-base/name->global-function "tvm.relay._save_param_dict"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

