(ns tvm-clj.compute.functional-tensor
  (:require [clojure.core.matrix.protocols :as mp]
            [tech.datatype.base :as dtype]
            [tech.compute.tensor :as ct]
            [tvm-clj.compute.tensor.functional-protocols :as fnp]))


(defn ecount
  ^long [item]
  (mp/element-count item))


(defn shape
  [item]
  (mp/get-shape item))


(defn get-datatype
  [item]
  (dtype/get-datatype item))


(defn select
  [item & args]
  (fnp/select ct/*stream* item args))


(defn transpose
  [item & args]
  (apply ct/transpose item args))


(defn static-cast
  [item new-dt]
  (fnp/static-cast ct/*stream* item new-dt))


(defn binary-op
  [lhs rhs op]
  (fnp/binary-op ct/*stream* lhs rhs op))

(defmacro define-binary-op
  [fn-name op]
  `(defn ~fn-name
     [lhs# rhs#]
     (binary-op lhs# rhs# ~op)))


(define-binary-op add :+)
(define-binary-op sub :-)
(define-binary-op mul :*)
(define-binary-op div :/)
(define-binary-op min :min)
(define-binary-op max :max)


(defn clamp
  [lhs min-val max-val]
  (-> (max lhs min-val)
      (min lhs max-val)))
