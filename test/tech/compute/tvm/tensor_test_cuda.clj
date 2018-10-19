(ns ^:cuda tech.compute.tvm.tensor-test-cuda
  (:require [tech.compute.verify.tensor :as vt]
            [tech.compute.tvm.test-utils :refer [def-all-dtype-test] :as cu]
            [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tech.compute.tvm :as tvm]))

(def-all-dtype-test assign-constant-cuda!
  (vt/assign-constant! (tvm/driver :cuda) *datatype*))

(def-all-dtype-test assign-cuda!
  (vt/assign-marshal (tvm/driver :cuda) *datatype*))
