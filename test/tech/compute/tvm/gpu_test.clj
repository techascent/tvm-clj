(ns tech.compute.tvm.gpu-test
  (:require [clojure.test :refer :all]
            [tech.compute.tvm.cpu-test :as cpu-test]
            [tech.compute.tvm.gpu]))


(deftest ^:cuda cuda-basic-add
  (cpu-test/test-add-fn :cuda))


(deftest opencl-basic-add
  (cpu-test/test-add-fn :opencl))
