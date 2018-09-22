(ns ^:cuda tvm-clj.compute.tensor-test-cuda
  (:require [tech.compute.verify.tensor :as vt]
            [tvm-clj.compute.test-utils :refer [def-all-dtype-test] :as cu]
            [tech.compute.verify.utils :refer [*datatype*] :as vu]
            [tvm-clj.compute.registry :as tvm-reg]))

(def-all-dtype-test assign-constant-cuda!
  (vt/assign-constant! (tvm-reg/get-driver :cuda) *datatype*))

(def-all-dtype-test assign-cuda!
  (vt/assign-marshal (tvm-reg/get-driver :cuda) *datatype*))
