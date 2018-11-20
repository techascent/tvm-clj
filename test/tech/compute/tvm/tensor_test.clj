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
            [clojure.test :refer :all]
            [tech.compute.tensor.math :as tm]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.defaults :as ct-defaults]
            [tech.resource :as resource]
            [clojure.core.matrix :as m]))


(def-all-dtype-test assign-constant-cpu!
  (vt/assign-constant! (tvm/driver :cpu) *datatype*))

(def-opencl-dtype-test assign-constant-opencl!
  (vt/assign-constant! (tvm/driver :opencl) :float32))

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


(defn tvm-gemm-upgrade
  "Test that tvm's gemm implementations correctly support transposed tensors.  We can't
  do this via the normal API because it normalizes tensors before they get to the tensor
  math level.  This is realy to benefit TVM and external parties using it's gemm
  functionality.  It has no benefit for tech.compute or the compute tvm backend."
  [driver datatype]
  (resource/with-resource-context
    (vt/tensor-default-context
     driver datatype
     (let [tens-a (ct/->tensor (partition 3 (range 9)))
           tens-b (ct/->tensor (partition 3 (repeat 9 2)))
           tens-c (ct/->tensor (partition 3 (repeat 9 10)))
           stream (ct-defaults/infer-stream {})]
       ;;For now, colstride and such are ignored.
       (tm/gemm! stream tens-c 0 false false 1 (ct/transpose tens-a [1 0]) 0 0 0
                 tens-b 0 0 0)
       ;;Same results as (gemm true false)
       (is (m/equals (ct/to-jvm tens-c)
                     [[18.0 18.0 18.0]
                      [24.0 24.0 24.0]
                      [30.0, 30.0, 30.0]]))

       ;;Same results as (gemm false false)
       (tm/gemm! stream tens-c 0 true false 1 (ct/transpose tens-a [1 0]) 0 0 0
                 tens-b 0 0 0)
       (is (m/equals (ct/to-jvm tens-c)
                     [[6.0 6.0 6.0]
                      [24.0 24.0 24.0]
                      [42.0 42.0 42.0]]))

       ;;Transposing C is illegal
       (is (thrown? Throwable (tm/gemm! stream (ct/transpose tens-c [1 0]) 0
                                        true false 1 tens-a 0 0 0
                                        tens-b 0 0 0)))))))


(deftest cpu-tvm-gemm-upgrade
  (tvm-gemm-upgrade (tvm/driver :cpu) *datatype*))


(deftest ^:cuda cuda-tvm-gemm-upgrade
  (tvm-gemm-upgrade (tvm/driver :cuda) *datatype*))


(deftest gc-rooted-tensors
  (testing "Ensure that gc-rooted tensors will *not* get cleaned up before their time"
    ;;Indexed tensor test tends to root out this stuff well, so we make it harder in
    ;;a way exactly designed to break the gc system if it is not correct.
    (vt/tensor-default-context
     (tvm/driver :opencl)
     :int64
     (let [sel-tens (ct/select (ct/->tensor (repeat 2 (partition 3 (range 9))))
                               :all :all [1 2])]
       ;;Note: there is no direct reference to the root data at all at this point.
       (System/gc)
       (Thread/sleep 100)
       ;;If the gc ran this will most likely crash.  In order to get the data into
       ;;a double array we have to copy it and that won't work if the source tensor
       ;;is freed.  This is the sort of subtle, non-deterministic thing that happens
       ;;when you mix gc-rooted things with external-to-gc things.
       (is (m/equals [1 2 4 5 7 8 1 2 4 5 7 8]
                     (vec (ct/to-double-array sel-tens))))))))
