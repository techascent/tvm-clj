(ns tvm-clj.compute.tensor.functional-protocols)


(defprotocol PFunctionalBackend
  (select [stream item dtype])
  (static-cast [stream item dtype])
  (binary-op [stream lhs rhs op]))
