(ns tvm-clj.tvm-test
  (:require [tvm-clj.ast :as ast]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tvm-clj.module :as module]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.functional :as dfn]
            [clojure.test :refer [deftest is]]))


(defn make-add-fn
  []
  ;;Default datatype of variable is integer
  (let [n (ast/variable "n")
        ;;Default datatype of placeholder is float32
        A (ast/placeholder [n] "A")
        B (ast/placeholder [n] "B")
        compute-op (ast/compute [n]
                                (ast/tvm-fn
                                 [i]
                                 (ast/add (ast/tget A [i])
                                          (ast/tget B [i])))
                                "C")
        C (first (ast/output-tensors compute-op))]
    {:schedule (schedule/create-schedule compute-op)
     :arguments [A B C]
     :compute-op compute-op}))


(deftest cpu-add
  (let [{:keys [schedule arguments compute-op]} (make-add-fn)
        _ (schedule/stage-cpu-injective schedule compute-op)
        module (compiler/build {"cpu_add" {:schedule schedule
                                           :arguments arguments}})
        add-fn (module/find-function module "cpu_add")
        tens-a (dtt/->tensor (range 10) :datatype :float32 :container-type :native-heap)
        tens-b (dtt/->tensor (range 10 20) :datatype :float32 :container-type :native-heap)
        tens-c (dtt/new-tensor [10] :datatype :float32 :container-type :native-heap)]
    (add-fn tens-a tens-b tens-c)
    (is (dfn/equals tens-c
                    (dfn/+ tens-a tens-b)))))


(deftest ^:cuda cuda-add
  (let [{:keys [schedule arguments compute-op]} (make-add-fn)
        _ (schedule/stage-gpu-injective schedule compute-op)
        module (compiler/build {"cuda_add" {:schedule schedule
                                            :arguments arguments
                                            :target "cuda"}})
        add-fn (module/find-function module "cuda_add")
        tens-a (dtt/->tensor (range 10) :datatype :float32 :container-type :native-heap)
        tens-b (dtt/->tensor (range 10 20) :datatype :float32 :container-type :native-heap)
        tens-c (dtt/new-tensor [10] :datatype :float32 :container-type :native-heap)]
    (add-fn tens-a tens-b tens-c)
    (is (dfn/equals tens-c
                    (dfn/+ tens-a tens-b)))))
