(ns tvm-clj.jna.fns.testing
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ErrorTest
(let [gfn* (delay (jna-base/name->global-function "testing.ErrorTest"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} context_test
(let [gfn* (delay (jna-base/name->global-function "testing.context_test"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} echo
(let [gfn* (delay (jna-base/name->global-function "testing.echo"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} nop
(let [gfn* (delay (jna-base/name->global-function "testing.nop"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} object_use_count
(let [gfn* (delay (jna-base/name->global-function "testing.object_use_count"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} test_check_eq_callback
(let [gfn* (delay (jna-base/name->global-function "testing.test_check_eq_callback"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} test_raise_error_callback
(let [gfn* (delay (jna-base/name->global-function "testing.test_raise_error_callback"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} test_wrap_callback
(let [gfn* (delay (jna-base/name->global-function "testing.test_wrap_callback"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

