(ns tvm-clj.api-test
  (:require [clojure.test :refer :all]
            [tvm-clj.api :as api]
            [tvm-clj.core :as c]
            [think.resource.core :as resource]
            [tech.datatype.core :as dtype]
            [tech.datatype.javacpp :as jcpp-dtype]
            [clojure.core.matrix :as m]))


(defn call-myadd-fn
  ^doubles [myadd-fn device device-id]
  (let [A (c/allocate-device-array [10] :float32 device device-id)
        B (c/allocate-device-array [10] :float32 device device-id)
        C (c/allocate-device-array [10] :float32 device device-id)
        ab-data (jcpp-dtype/make-pointer-of-type :float32 (range 10))
        result-ptr (jcpp-dtype/make-pointer-of-type :float32 10)
        result-ary (double-array 10)]
    (c/copy-to-array! ab-data A (* 10 Float/BYTES))
    (c/copy-to-array! ab-data B (* 10 Float/BYTES))
    (myadd-fn A B C)
    (c/copy-from-array! C result-ptr (* 10 Float/BYTES))
    (dtype/copy! result-ptr 0 result-ary 0 10)
    result-ary))


(defn create-myadd-fn
  [build-target]
  (let [n (api/variable "n")
        A (api/placeholder [n] "A")
        B (api/placeholder [n] "B")
        compute-op (api/compute [n] (api/tvm-fn
                                     [i]
                                     (api/add (api/tget A [i])
                                              (api/tget B [i])))
                                "C")
        C (first (api/output-tensors compute-op))
        schedule (api/create-schedule compute-op)]
    (if (= :cpu build-target)
      (api/stage-cpu-injective schedule compute-op)
      (api/stage-gpu-injective schedule compute-op))
    (-> (api/schedules->fns [{:name :myadd
                              :arglist [A B C]
                              :schedule schedule}]
                            :target-name build-target)
        (get-in [:fn-map :myadd]))))


(deftest add-cpu
  (resource/with-resource-context
    (let [myadd-fn (create-myadd-fn :cpu)]
      (is (m/equals (m/add (m/array (range 10)) (m/array (range 10)))
                    (vec (call-myadd-fn myadd-fn :cpu 0)))))))


(deftest add-opencl
  (resource/with-resource-context
    (let [myadd-fn (create-myadd-fn :opencl)]
      (is (m/equals (m/add (m/array (range 10)) (m/array (range 10)))
                    (vec (call-myadd-fn myadd-fn :opencl 0)))))))


(deftest ^:cuda add-cuda
  (resource/with-resource-context
    (let [myadd-fn (create-myadd-fn :cuda)]
      (is (m/equals (m/add (m/array (range 10)) (m/array (range 10)))
                    (vec (call-myadd-fn myadd-fn :cuda 0)))))))
