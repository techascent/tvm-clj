(ns tvm-clj.core
  (:require [clojure.reflect :as reflect])
  (:import [ml.dmlc.tvm TVMContext
            Function Base LibInfo]
           [java.util ArrayList]))



(defn list-global-function-names
  []
  (let [fun-data (->> (.getDeclaredMethods Function)
                      (filter #(= "listGlobalFuncNames"
                                  (.getName %)))
                      first)]
    (.setAccessible fun-data true)
    (->> (.invoke fun-data Function (object-array 0))
         seq
         vec)))
