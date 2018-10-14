(ns tvm-clj.compute.tensor-test
  (:require [tech.compute.verify.tensor :as vt]
            [tvm-clj.compute.test-utils :refer [def-all-dtype-test
                                                def-opencl-dtype-test] :as cu]
            [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tvm-clj.compute.registry :as tvm-reg]
            [tvm-clj.compute.cpu]
            [tvm-clj.compute.gpu]
            [tvm-clj.compute.tensor-math]
            [clojure.test :refer :all]))



(def-all-dtype-test assign-constant-cpu!
  (vt/assign-constant! (tvm-reg/get-driver :cpu) *datatype*))

(def-opencl-dtype-test assign-constant-opencl!
  (vt/assign-constant! (tvm-reg/get-driver :opencl) *datatype*))


(def-all-dtype-test assign-cpu!
  (vt/assign-marshal (tvm-reg/get-driver :cpu) *datatype*))

(def-opencl-dtype-test assign-opencl!
  (vt/assign-marshal (tvm-reg/get-driver :opencl) *datatype*))
