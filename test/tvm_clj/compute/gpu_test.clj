(ns tvm-clj.compute.gpu-test
  (:require [tvm-clj.compute.gpu]
            [tvm-clj.compute.cpu-test :as cpu-test]
            [tvm-clj.compute.registry :as tvm-reg]
            [tvm-clj.core :as tvm-core]
            [tvm-clj.api-test :as api-test]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [tech.resource :as resource]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(deftest ^:cuda cuda-basic-add
  (resource/with-resource-context
    (cpu-test/test-add-fn (tvm-reg/get-device :cuda 0)
                          (api-test/create-myadd-fn :cuda))))


(deftest opencl-basic-add
  (resource/with-resource-context
    (cpu-test/test-add-fn (tvm-reg/get-device :opencl 0)
                          (api-test/create-myadd-fn :opencl))))
