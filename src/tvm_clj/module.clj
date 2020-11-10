(ns tvm-clj.module
  "Once user's have a compiled a module, the then can query the module
  for the functions within.  Functions returned take only things convertible
  to TVM nodes such as scalars and tensors and the result buffer must be
  passed in."
  (:require [tvm-clj.impl.module :as mod-impl]
            [tvm-clj.impl.dl-tensor :as dl-tensor]))


(defn find-function
  "Find a function in module.  Failure causes an exception."
  [module fn-name]
  (mod-impl/get-module-function module fn-name false))
