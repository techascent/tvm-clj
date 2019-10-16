(ns tech.libs.tvm.gpu-test
  (:require [clojure.test :refer :all]
            [tech.libs.tvm.cpu-test :as cpu-test]
            [tech.libs.tvm.gpu]))


(deftest ^:cuda cuda-basic-add
  (cpu-test/test-add-fn :cuda))


(deftest opencl-basic-add
  (cpu-test/test-add-fn :opencl))
