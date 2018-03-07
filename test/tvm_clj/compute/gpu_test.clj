(ns tvm-clj.compute.gpu-test
  (:require [tvm-clj.compute.gpu]
            [tvm-clj.compute.cpu-test :as cpu-test]
            [tvm-clj.compute.base :as base]
            [tvm-clj.core :as tvm-core]
            [tvm-clj.api-test :as api-test]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [think.resource.core :as resource]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(deftest cuda-basic-add
  (resource/with-resource-context
    (cpu-test/test-add-fn (base/get-device :cuda 0)
                          (api-test/create-myadd-fn-gpu :cuda))))


(deftest opencl-basic-add
  (resource/with-resource-context
    (cpu-test/test-add-fn (base/get-device :opencl 0)
                          (api-test/create-myadd-fn-gpu :opencl))))
