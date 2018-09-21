(ns tvm-clj.compute.test-utils
  (:require [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [clojure.test :refer [deftest]]))


(def all-dtype-table
  {:float64 "-d"
   :float32 "-f"
   :int64 "-l"
   :uint64 "-ul"
   :int32 "-i"
   :uint32 "-ui"
   :int16 "-s"
   :uint16 "-us"
   :int8 "-b"
   :uint8 "-b"})


(def opencl-dtype-table
  ;;Vanilla opencl doesn't support 64 bit computing
  (dissoc all-dtype-table :float64))


(defmacro def-dtype-tests
  [test-name dtype-table & body]
  `(do
     ~@(for [[dtype suffix] dtype-table]
         (let [test-name (symbol (str test-name suffix))]
           `(deftest ~test-name
              (with-bindings {#'*datatype* ~dtype}
                ~@body))))))


(defmacro def-all-dtype-test
  [test-name & body]
  `(def-dtype-tests ~test-name ~all-dtype-table ~@body))


(defmacro def-opencl-dtype-test
  [test-name & body]
  `(def-dtype-tests ~test-name ~opencl-dtype-table ~@body))
