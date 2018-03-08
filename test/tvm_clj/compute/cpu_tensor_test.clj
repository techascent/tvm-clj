(ns tvm-clj.compute.cpu-tensor-test
  (:require [tech.compute.verify.tensor :as vt]
            [tvm-clj.compute.test-utils :as cu]
            [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tvm-clj.compute.cpu :as tvm-cpu]
            [tvm-clj.compute.base :as base]
            [tvm-clj.base :as root]
            [tech.compute.driver :as drv]
            [tech.compute.tensor :as ct]
            [tech.datatype.base :as dtype]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tvm-clj.compute.cpu-tensor-math]
            [clojure.test :refer :all]))



(cu/def-all-dtype-test assign-constant!
  (vt/assign-constant! (base/get-driver :cpu) *datatype*))
