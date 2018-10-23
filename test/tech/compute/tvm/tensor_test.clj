(ns tech.compute.tvm.tensor-test
  (:require [tech.compute.verify.tensor :as vt]
            [tech.compute.tvm.test-utils :refer [def-all-dtype-test
                                                 def-opencl-dtype-test] :as cu]
            [tech.compute.verify.utils :refer [*datatype*
                                               def-all-dtype-exception-unsigned
                                               def-double-float-test
                                               def-int-long-test] :as vu]
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

(def-all-dtype-exception-unsigned binary-constant-op
  (vt/binary-constant-op (tvm/driver :cpu) *datatype*))


(def-all-dtype-exception-unsigned binary-op
  (vt/binary-op (tvm/driver :cpu) *datatype* ))


(def-all-dtype-test unary-op
  (vt/unary-op (tvm/driver :cpu) *datatype*))


(def-all-dtype-test channel-op
  (vt/channel-op (tvm/driver :cpu) *datatype*))


(def-double-float-test gemm
  (vt/gemm (tvm/driver :cpu) *datatype*))


(def-all-dtype-exception-unsigned ternary-op-select
  (vt/ternary-op-select (tvm/driver :cpu) *datatype*))


(def-all-dtype-test unary-reduce
  (vt/unary-reduce (tvm/driver :cpu) *datatype*))


(def-all-dtype-test transpose
  (vt/transpose (tvm/driver :cpu) *datatype*))


(def-int-long-test mask
  (vt/mask (tvm/driver :cpu) *datatype*))


(def-all-dtype-test select
  (vt/select (tvm/driver :cpu) *datatype*))


(def-all-dtype-test select-with-persistent-vectors
  (vt/select-with-persistent-vectors (tvm/driver :cpu) *datatype*))


(def-all-dtype-test select-transpose-interaction
  (vt/select-transpose-interaction (tvm/driver :cpu) *datatype*))


;;Note that this is not a float-double test.
(deftest rand-operator
  (vt/rand-operator (tvm/driver :cpu) :float32))


(def-all-dtype-test indexed-tensor
  (vt/indexed-tensor (tvm/driver :cpu) *datatype*))


(def-double-float-test magnitude-and-mag-squared
  (vt/magnitude-and-mag-squared (tvm/driver :cpu) *datatype*))


(def-double-float-test constrain-inside-hypersphere
  (vt/constrain-inside-hypersphere (tvm/driver :cpu) *datatype*))
