(ns tvm-clj.compute.tensor-test
  (:require [tech.compute.verify.tensor :as vt]
            [tvm-clj.compute.test-utils :refer [def-all-dtype-test] :as cu]
            [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tvm-clj.compute.cpu :as tvm-cpu]
            [tvm-clj.compute.base :as base]
            [tvm-clj.base :as root]
            [tech.compute.driver :as drv]
            [tech.compute.tensor :as ct]
            [tech.datatype.base :as dtype]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tvm-clj.compute.tensor-math]
            [clojure.test :refer :all]))



(def-all-dtype-test assign-constant-cpu!
  (vt/assign-constant! (base/get-driver :cpu) *datatype*))

(def-all-dtype-test assign-constant-cuda!
  (vt/assign-constant! (base/get-driver :cuda) *datatype*))

(def-all-dtype-test assign-constant-opencl!
  (vt/assign-constant! (base/get-driver :opencl) *datatype*))


(def-all-dtype-test assign-cpu!
  (vt/assign-marshal (base/get-driver :cpu) *datatype*))

(def-all-dtype-test assign-cuda!
  (vt/assign-marshal (base/get-driver :cuda) *datatype*))

(def-all-dtype-test assign-opencl!
  (vt/assign-marshal (base/get-driver :opencl) *datatype*))
