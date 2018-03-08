(ns tvm-clj.compute.test-utils
  (:require [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [clojure.test :refer [deftest]]))


(defmacro def-all-dtype-test
  [test-name & body]
  (let [double-test-name (str test-name "-d")
        float-test-name (str test-name "-f")
        long-test-name (str test-name "-l")
        int-test-name (str test-name "-i")
        short-test-name (str test-name "-s")
        byte-test-name (str test-name "-b")
        ulong-test-name (str test-name "-ul")
        uint-test-name (str test-name "-ui")
        ushort-test-name (str test-name "-us")
        ubyte-test-name (str test-name "-ub")]
   `(do
      (deftest ~(symbol double-test-name)
        (with-bindings {#'*datatype* :float64}
          ~@body))
      (deftest ~(symbol float-test-name)
        (with-bindings {#'*datatype* :float32}
          ~@body))
      (deftest ~(symbol long-test-name)
        (with-bindings {#'*datatype* :int64}
          ~@body))
      (deftest ~(symbol int-test-name)
        (with-bindings {#'*datatype* :int32}
          ~@body))
      (deftest ~(symbol short-test-name)
        (with-bindings {#'*datatype* :int16}
          ~@body))
      (deftest ~(symbol byte-test-name)
        (with-bindings {#'*datatype* :int8}
          ~@body))
      (deftest ~(symbol ulong-test-name)
        (with-bindings {#'*datatype* :uint64}
          ~@body))
      (deftest ~(symbol uint-test-name)
        (with-bindings {#'*datatype* :uint32}
          ~@body))
      (deftest ~(symbol ushort-test-name)
        (with-bindings {#'*datatype* :uint16}
          ~@body))
      (deftest ~(symbol ubyte-test-name)
        (with-bindings {#'*datatype* :uint8}
          ~@body)))))
