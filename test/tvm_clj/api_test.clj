(ns tvm-clj.api-test
  (:require [clojure.test :refer :all]
            [tvm-clj.api :as api]
            [tvm-clj.core :as c]
            [think.resource.core :as resource]
            [tech.datatype.core :as dtype]
            [tech.javacpp-datatype :as jcpp-dtype]
            [clojure.core.matrix :as m]))


(deftest add-cpu
  (resource/with-resource-context
    (let [myadd-fn (let [n (api/variable "n")
                         A (api/placeholder [n] :name "A")
                         B (api/placeholder [n] :name "B")
                         compute-op (api/compute [n] (api/tvm-fn
                                                      [i]
                                                      (api/add (api/tget A [i])
                                                               (api/tget B [i])))
                                                 :name "C")
                         C (first (api/output-tensors compute-op))
                         schedule (api/create-schedule compute-op)
                         fn-name "myadd"
                         lowered-fn (api/schedule->lowered-function schedule [A B C] api/default-build-config :name fn-name)
                         module (api/lowered-functions->module [lowered-fn] api/default-build-config)]
                     (c/get-module-function module fn-name))
          A (c/allocate-device-array [10] :float32 :cpu 0)
          B (c/allocate-device-array [10] :float32 :cpu 0)
          C (c/allocate-device-array [10] :float32 :cpu 0)
          ab-data (jcpp-dtype/make-pointer-of-type :float (range 10))
          result-ptr (jcpp-dtype/make-pointer-of-type :float 10)
          result-ary (double-array 10)]
      (c/copy-to-array! ab-data A (* 10 Float/BYTES))
      (c/copy-to-array! ab-data B (* 10 Float/BYTES))
      (c/call-function myadd-fn A B C)
      (c/copy-from-array! C result-ptr (* 10 Float/BYTES))
      (dtype/copy! result-ptr 0 result-ary 0 10)
      (is (m/equals (m/add (m/array (range 10)) (m/array (range 10)))
                    (vec result-ary))))))


(defn add-gpu
  []
  (let [n (api/variable "n")
        A (api/placeholder [n] :name "A")
        B (api/placeholder [n] :name "B")
        compute-op (api/compute [n] (api/tvm-fn
                                     [i]
                                     (api/add (api/tget A [i])
                                              (api/tget B [i])))
                                :name "C")
        C (first (api/output-tensors compute-op))
        schedule (api/create-schedule compute-op)
        compute-stage (get-in schedule [:stage_map compute-op])
        [bx tx] (api/split-stage-by-factor compute-stage (get-in compute-op [:axis 0]) 64)]

    (api/stage-bind compute-stage bx (api/name->thread-axis-iterator "blockIdx.x"))
    (api/stage-bind compute-stage tx (api/name->thread-axis-iterator "threadIdx.x"))
    [bx tx]))
