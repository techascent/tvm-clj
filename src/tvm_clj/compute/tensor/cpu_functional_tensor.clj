(ns tvm-clj.compute.tensor.cpu-functional-tensor
  (:require [tvm-clj.compute.tensor.functional-protocols :as fp]
            [tech.compute.cpu.driver :as cpu-driver]
            [tech.compute.tensor :as ct]
            [tech.compute.cpu.tensor-math]
            [tech.datatype.core :as dtype-core]
            [tech.compute.driver :as drv])
  (:import [tech.compute.cpu.driver CPUStream]
           [tech.compute.tensor Tensor]))



(extend-type CPUStream
  fp/PFunctionalBackend
  ;;Because we have defined a slightly different language the base tensor
  ;;won't work out of the box.  The CPU system expects a defined format for
  ;;select
  (select [stream item args]
    (apply ct/select item
           (map (fn [arg]
                  (cond
                    (keyword? arg)
                    arg
                    (sequential? arg)
                    (let [int-data (int-array arg)]
                      (ct/->Tensor (drv/get-device stream)
                                   {:shape [(alength int-data)]
                                    :strides [1]}
                                   (dtype-core/->view (int-array arg))))))
                args)))
  (transpose [stream item reorder-vec]
    (ct/transpose item reorder-vec))

  (static-cast [stream item dtype dest-shape]
    (let [retval (ct/new-tensor (or dest-shape (ct/shape item))
                                :datatype dtype :init-value nil)]
      (ct/assign! retval item)
      retval))

  (binary-op [stream lhs rhs op dest-shape]
    (let [primary-tensor (->> (filter #(instance? Tensor %) [lhs rhs])
                              first)
          res-shape (or dest-shape (ct/shape primary-tensor))
          retval (ct/new-tensor dest-shape
                                :datatype (ct/get-datatype primary-tensor)
                                :init-value nil)]
      (ct/binary-op! retval 1.0 lhs 1.0 rhs op)
      retval)))
