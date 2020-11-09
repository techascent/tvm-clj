(ns tvm-clj.impl.fns.testing
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private ErrorTest-fnptr* (delay (base/name->global-function "testing.ErrorTest")))
(defn ErrorTest
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.ErrorTest"}
   (apply base/call-function @ErrorTest-fnptr* args)))

(defonce ^:private context_test-fnptr* (delay (base/name->global-function "testing.context_test")))
(defn context_test
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.context_test"}
   (apply base/call-function @context_test-fnptr* args)))

(defonce ^:private echo-fnptr* (delay (base/name->global-function "testing.echo")))
(defn echo
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.echo"}
   (apply base/call-function @echo-fnptr* args)))

(defonce ^:private nop-fnptr* (delay (base/name->global-function "testing.nop")))
(defn nop
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.nop"}
   (apply base/call-function @nop-fnptr* args)))

(defonce ^:private object_use_count-fnptr* (delay (base/name->global-function "testing.object_use_count")))
(defn object_use_count
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.object_use_count"}
   (apply base/call-function @object_use_count-fnptr* args)))

(defonce ^:private test_check_eq_callback-fnptr* (delay (base/name->global-function "testing.test_check_eq_callback")))
(defn test_check_eq_callback
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.test_check_eq_callback"}
   (apply base/call-function @test_check_eq_callback-fnptr* args)))

(defonce ^:private test_raise_error_callback-fnptr* (delay (base/name->global-function "testing.test_raise_error_callback")))
(defn test_raise_error_callback
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.test_raise_error_callback"}
   (apply base/call-function @test_raise_error_callback-fnptr* args)))

(defonce ^:private test_wrap_callback-fnptr* (delay (base/name->global-function "testing.test_wrap_callback")))
(defn test_wrap_callback
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "testing.test_wrap_callback"}
   (apply base/call-function @test_wrap_callback-fnptr* args)))

