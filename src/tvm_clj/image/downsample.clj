(ns tvm-clj.image.downsample
  (:require [tech.compute.tensor :as ct]
            [tech.compute.driver :as drv]
            [tvm-clj.compute.cpu]
            [tvm-clj.compute.tensor-math]
            [clojure.core.matrix :as m]
            [tech.datatype.base :as dtype]
            [tvm-clj.api :as api]
            [tvm-clj.core :as c]
            [tvm-clj.compute.registry :as registry]
            [tech.compute.verify.tensor :as verify-tensor]))


;;uint8 input/output tensors.
(defn- n-img-shape
  [shape-vec]
  (case (count shape-vec)
    2 (vec (concat [1] shape-vec [1]))
    3 (vec (concat [1] shape-vec))
    4 shape-vec))

(defn- to-long-round
  ^long [value]
  (long (Math/round value)))


(defn- to-long-ceil
  ^long [value]
  (long (Math/ceil value)))


(defn convolve-filter
  "Assuming dense image tensors, convolve a 2d kernel that can be described by the
  cartesian join(*) of 2 1 dimensional kernel vectors over an image of indeterminate
  dimensions to produce a result of indeterminate dimensions.  Input is clamped-to-edge,
  intermediate type is uint16, kernels are float numbers, output is same datatype
  as input."
  [img-dtype]
  (let [in-width (api/variable "in_width")
        in-height (api/variable "in_height")
        out-width (api/variable "out_width")
        out-height (api/variable "out_height")
        n-chans (api/variable "n_channels")
        input (api/placeholder [in-height in-width n-chans] :dtype img-dtype)
        clamp (api/tvm-fn
               [value val_min val_max]
               (-> (api/min value val_max)
                   (api/max val_min)))

        ;;Clamp input to edge and convert to float.
        input-to-ushort (api/tvm-fn
                         [y x c]
                         (api/static-cast :uint16
                                          (api/tget input
                                                    [(clamp y 0 (api/sub in-height 1))
                                                     (clamp x 0 (api/sub in-width 1))
                                                     c])))
        k-width (api/variable "k_width")
        k-height (api/variable "k_height")
        ;; kern_x (api/placeholder [k-width] :dtype :float32)
        ;; kern_y (api/placeholder [k-height] :dtype :float32)
        kern_x_axis (api/iteration-variable [0 k-width] "red_x" :communicative-reduce)
        kern_y_axis (api/iteration-variable [0 k-height] "red_y" :communicative-reduce)
        input-idx (api/tvm-fn
                   [dest-idx kern-size kern-idx]
                   (api/add dest-idx
                            (api/sub kern-idx
                                     (api/div
                                      (api/sub kern-size 1)
                                      2))))

        compute-op (api/compute [out-height out-width n-chans]
                                (api/tvm-fn
                                 [y x c]
                                 (api/commutative-reduce
                                  (api/tvm-fn [lhs rhs] (api/add lhs rhs))
                                  (api/const 0 :dtype :uint16)
                                  :uint16
                                  (input-to-ushort (input-idx y k-height kern_y_axis)
                                                   (input-idx x k-width kern_x_axis)
                                                   c)
                                  [kern_y_axis kern_x_axis])))
        output (first (api/output-tensors compute-op))]
    {:bind-list [input output k-width k-height]
     :fn! compute-op}))


(defn test-convolve-filter
  []
  (verify-tensor/tensor-context
   (registry/get-driver :cpu)
   :uint8
   (let [device drv/*current-compute-device*
         driver (drv/get-driver device)
         {:keys [bind-list fn!]} (convolve-filter :uint8)
         schedule (registry/schedule-injective driver fn!)
         lowered-fn (api/schedule->lowered-function
                     schedule bind-list
                     api/default-build-config
                     :name "test_fn")
         module (registry/->module driver [lowered-fn])
         actual-fn (c/get-module-function module "test_fn")
         src-tensor (ct/->tensor (->> (range (* 2 2 2))
                                      (partition 2)
                                      (partition 2)))
         dst-tensor (ct/new-tensor [2 1 2] :datatype :uint16 :init-value 0)]

     (clojure.pprint/pprint (->> (ct/to-float-array dst-tensor)
                                 (partition 2)
                                 (partition 4)))
     (c/call-function actual-fn src-tensor dst-tensor 2 1)
     (clojure.pprint/pprint (->> (ct/to-float-array dst-tensor)
                                 (partition 2)
                                 (partition 4)))))
  nil)
