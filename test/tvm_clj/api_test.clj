(ns tvm-clj.api-test
  (:require [clojure.test :refer :all]
            [tvm-clj.api :as api]
            [tvm-clj.core :as c]
            [think.resource.core :as resource]))


(defn add-cpu
  []
  (let [n (api/variable "n")
        A (api/placeholder [n] :name "A")
        B (api/placeholder [n] :name "B")
        compute-op (api/compute [n] (api/tvm-fn [i] (api/add (api/tget A [i])
                                                             (api/tget B [i])))
                                :name "C")
        C (first (api/output-tensors compute-op))
        schedule (api/create-schedule compute-op)]
    (api/lower schedule [A B C] api/default-build-config :name "fadd")))
