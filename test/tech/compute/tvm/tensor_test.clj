(ns tech.compute.tvm.tensor-test
  (:require [tech.compute.verify.tensor :as vt]
            [tech.compute.tvm.test-utils :refer [def-all-dtype-test
                                                 def-opencl-dtype-test] :as cu]
            [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tech.compute.tvm :as tvm]
            [tech.compute.tvm.cpu]
            [tech.compute.tvm.gpu]
            [tech.compute.tvm.tensor-math]
            [clojure.test :refer :all]))


(def-all-dtype-test assign-constant-cpu!
  (vt/assign-constant! (tvm/driver :cpu) *datatype*))

(def-opencl-dtype-test assign-constant-opencl!
  (vt/assign-constant! (tvm/driver :opencl) *datatype*))

(def-all-dtype-test assign-cpu!
  (vt/assign-marshal (tvm/driver :cpu) *datatype*))

(def-opencl-dtype-test assign-opencl!
  (vt/assign-marshal (tvm/driver :opencl) *datatype*))
