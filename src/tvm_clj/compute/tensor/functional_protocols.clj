(ns tvm-clj.compute.tensor.functional-protocols)


(defprotocol PFunctionalBackend
  (select [stream item args])
  (transpose [stream item reorder-vec])
  (static-cast [stream item dtype dest-shape])
  (binary-op [stream lhs rhs op dest-shape]))
