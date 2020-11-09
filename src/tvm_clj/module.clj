(ns tvm-clj.module
  (:require [tvm-clj.impl.module :as mod-impl]
            [tvm-clj.impl.dl-tensor :as dl-tensor]))


(defn find-function
  [module fn-name]
  (mod-impl/get-module-function module fn-name false))
