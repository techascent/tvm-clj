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
  [item reorder-vec]
  (fnp/transpose ct/*stream* item reorder-vec))


(defn static-cast
  [item new-dt & {:keys [dest-shape]}]
  (fnp/static-cast ct/*stream* item new-dt (or dest-shape (mp/get-shape item))))


(defn binary-op
  [lhs rhs op dest-shape]
  (fnp/binary-op ct/*stream* lhs rhs op (or dest-shape (mp/get-shape lhs))))

(defmacro define-binary-op
  [fn-name op]
  (let [fn-name-r (symbol (str (name fn-name) "-r"))
        dest-shape (symbol "dest-shape")
        lhs (symbol "lhs")
        rhs (symbol "rhs")]
    `(do
       (defn ~fn-name
         [~lhs ~rhs & {:keys [~dest-shape]}]
         (binary-op ~lhs ~rhs ~op ~dest-shape))
       (defn ~fn-name-r
         [~rhs ~lhs & {:keys [~dest-shape]}]
         (binary-op ~lhs ~rhs ~op ~dest-shape)))))


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
