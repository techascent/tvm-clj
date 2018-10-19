(ns tech.compute.tvm.test-utils
  (:require [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tech.datatype.java-unsigned :as unsigned]
            [clojure.test :refer [deftest]]))

(def all-dtype-table (set unsigned/datatypes))

(def opencl-dtype-table (disj all-dtype-table :float64))


(defmacro def-all-dtype-test
  [test-name & body]
  `(vu/datatype-list-tests ~all-dtype-table ~test-name ~@body))


(defmacro def-opencl-dtype-test
  [test-name & body]
  `(vu/datatype-list-tests ~opencl-dtype-table ~test-name ~@body))
