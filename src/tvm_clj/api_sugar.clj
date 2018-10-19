(ns tvm-clj.api-sugar
  "Operators and bindings to make the clojure code look and work like the python
tvm bindings.  This file is purely syntax sugar."
  (:require [tvm-clj.api :as api]
            [tvm-clj.tvm-bindings :as bindings]))

(set! *warn-on-reflection* true)


(defmacro defbinop
  [op-symbol scalar-fn api-fn]
  `(defn ~op-symbol
     ([lhs# rhs#]
      (if (or (bindings/is-node-handle? lhs#)
              (bindings/is-node-handle? rhs#))
        (~api-fn lhs# rhs#)
        (~scalar-fn lhs# rhs#)))
     ([lhs# rhs# arg# & args#]
      (apply ~op-symbol
             (~op-symbol lhs# rhs#)
             arg# args#))))

(defmacro defunop
  [op-symbol scalar-fn api-fn]
  `(defn ~op-symbol
     [lhs#]
     (if (bindings/is-node-handle? lhs#)
       (~api-fn lhs#)
       (~scalar-fn lhs#))))

(defmacro defapionlyunary
  [op-symbol api-fn]
  `(defn op-symbol
     [lhs#]
     (when-not (bindings/is-node-handle? lhs#)
       (throw (ex-info "Operation doesn't have a java equivalent"
                       {:operation (str ~op-symbol)})))
     (api-fn lhs#)))

(defbinop + clojure.core/+ api/add)
(defbinop - clojure.core/- api/sub)
(defbinop * clojure.core/* api/mul)
(defbinop / clojure.core// api/div)
(defbinop rem clojure.core/rem api/mod)
(defbinop = clojure.core/= api/eq)
(defbinop min clojure.core/min api/min)
(defbinop max clojure.core/max api/max)
(defbinop pow Math/pow api/power)
(defunop exp Math/exp api/exp)
(defunop tanh Math/tanh api/tanh)
(defunop sigmoid #(/ 1.0
                     (+ 1.0 (Math/exp (- %)))) api/sigmoid)
(defunop log Math/log api/log)
(defunop sqrt Math/sqrt api/sqrt)
(defunop floor Math/floor api/floor)
(defunop ceil Math/ceil api/ceil)
(defunop abs #(Math/abs (double %)) api/abs)
(defunop round #(Math/round (double %)) api/round)
(defunop trunc #(if (> % 0)
                  (Math/floor %)
                  (Math/ceil %)) api/trunc)
(defunop popcount #(Long/bitCount (long %)) api/popcount)
(def tvar api/variable)
(def const api/const)
(def placeholder api/placeholder)
(def cast api/cast)


(defn compute
  "Returns the output tensor or a vector instead of the operation.
You can recover the operation from any output's op member."
  [dims fn name & args]
  (let [target-op (apply api/compute dims fn name args)
        output-tensors (api/output-tensors target-op)]
    (if (= 1 (count output-tensors))
      (first output-tensors)
      output-tensors)))


(defmacro lambda
  [arglist & body]
  `(api/tvm-fn ~arglist ~@body))


(defmacro tif
  [bool-stmt true-stmt false-stmt]
  `(let [bool-arg# ~bool-stmt]
     (if (bindings/is-node-handle? bool-arg#)
       (api/select bool-arg# ~true-stmt ~false-stmt)
       (if bool-arg#
         ~true-stmt
         ~false-stmt))))


(defmacro tlet
  [expr-pairs body]
  `(api/tvm-let ~expr-pairs ~body))
