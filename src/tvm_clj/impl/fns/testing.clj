(ns tvm-clj.jna.fns.testing
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "testing.ErrorTest"))]
  (defn ErrorTest
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.ErrorTest"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.context_test"))]
  (defn context_test
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.context_test"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.echo"))]
  (defn echo
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.echo"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.nop"))]
  (defn nop
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.nop"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.object_use_count"))]
  (defn object_use_count
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.object_use_count"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.test_check_eq_callback"))]
  (defn test_check_eq_callback
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.test_check_eq_callback"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.test_raise_error_callback"))]
  (defn test_raise_error_callback
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.test_raise_error_callback"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "testing.test_wrap_callback"))]
  (defn test_wrap_callback
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "testing.test_wrap_callback"}
     (apply jna-base/call-function @gfn* args))))

