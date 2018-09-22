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


(defn- clamp
  [value val_min val_max]
  (-> (api/min value val_max)
      (api/max val_min)))


(defn- read-clamped-f32
  [img in-height in-width y x c]
  (api/static-cast :float32
                   (api/tget img
                             [(clamp y 0 (api/sub in-height 1))
                              (clamp x 0 (api/sub in-width 1))
                              c])))


(defn convolve-filter
  "Assuming dense image tensors, convolve a 2d kernel that can be described by the
  cartesian join(*) of two 1 dimensional kernel vectors over an image of indeterminate
  dimensions to produce a result of indeterminate dimensions.  Input is clamped-to-edge,
  intermediate type is uint16, kernels are float numbers, output is same datatype
  as input.  Your convolve kernel needs to be linearly seperable in x and y."
  [img-dtype driver]
  (let [in-width (api/variable "in_width")
        in-height (api/variable "in_height")
        out-width (api/variable "out_width")
        out-height (api/variable "out_height")
        n-chans (api/variable "n_channels")
        input (api/placeholder [in-height in-width n-chans] :dtype img-dtype)

        ;;Clamp input to edge and convert to float.
        rea-to-f32 (api/tvm-fn
                      [input y x c]
)
        k-width (api/variable "k_width")
        k-height (api/variable "k_height")
        kern_x (api/placeholder [k-width] :dtype :float32)
        kern_y (api/placeholder [k-height] :dtype :float32)
        kern_x_axis (api/iteration-variable [0 k-width] "red_x" :communicative-reduce)
        kern_y_axis (api/iteration-variable [0 k-height] "red_y" :communicative-reduce)
        input-idx (api/tvm-fn
                   [dest-idx kern-size kern-idx]
                   (api/add dest-idx
                            (api/sub kern-idx
                                     (api/div
                                      kern-size
                                      (api/const 2 :dtype :int32)))))

        compute-op (api/compute [out-height out-width n-chans]
                                (api/tvm-fn
                                 [y x c]
                                 (api/static-cast
                                  img-dtype
                                  (api/commutative-reduce
                                   (api/tvm-fn
                                    [lhs rhs y_mul x_mul]
                                    (api/add lhs (-> (api/mul rhs y_mul)
                                                     (api/mul x_mul))))
                                   (api/const 0 :dtype :float32)
                                   :float32
                                   [(read-clamped-f32
                                     input in-height in-width
                                     (input-idx y k-height kern_y_axis)
                                     (input-idx x k-width kern_x_axis)
                                     c)
                                    (api/tget kern_y [kern_y_axis])
                                    (api/tget kern_x [kern_x_axis])]
                                   [kern_y_axis kern_x_axis]))))
        output (first (api/output-tensors compute-op))
        schedule (registry/schedule-injective driver compute-op)
        fn-name (str "convolve_" (name img-dtype))
        lowered-fn (api/schedule->lowered-function
                    schedule [input output kern_y kern_x]
                    api/default-build-config
                    :name fn-name)
        module (registry/->module driver [lowered-fn])]
    (c/get-module-function module fn-name)))


(defn- bilinear-filter-pixel-size
  ^long [in-size out-size]
  (let [temp (/ (double in-size)
                (double out-size))]
    (long
     (if (= (Math/floor temp)
            temp)
       temp
       ;;If it is fractional then it could overlap on either side
       ;;at the same time.
       (Math/floor (+ temp 2))))))

(defn- bilinear-addr-mul
  "Clojure version to calculate src address and to calculate multiplier"
  [dst-pixel ratio kern-pix-idx]
  (let [src-start (double (+ (* dst-pixel ratio)))
        src-end (+ src-start ratio)
        src-pixel (Math/floor (+ src-start kern-pix-idx))]
    {:src-pixel src-pixel
     :src-start src-start
     :src-end src-end
     :src-mul (if (= 0 kern-pix-idx)
                (- (+ 1.0 src-pixel) src-start)
                (Math/min 1.0 (- src-end src-pixel)))}))

(defn- pixel-mul
  [start-pix start item-range item-idx]
  (api/select (api/eq (api/const 0 :dtype :int32)
                      item-idx)
              (api/sub (api/add (api/const 1.0 :dtype :float32)
                                (api/static-cast :float32 start-pix))
                       start)
              (clamp (api/sub (api/add start item-range)
                              (api/static-cast :float32 start-pix))
                     (float 0.0)
                     (float 1.0))))


(defn- read-src-pixel
  [input in-height in-width
   dst-y dst-x dst-c
   kern-y kern-x
   y-ratio x-ratio]
  (api/tvm-let
   [left (api/mul dst-x x-ratio)
    top (api/mul dst-y y-ratio)
    ;;TVM has no floor function so we use the old trick of casting.
    ;;I am sure this is quite slow.
    left-pix (api/add (api/static-cast :int32 left) kern-x)
    top-pix (api/add (api/static-cast :int32 top) kern-y)
    left-mul (pixel-mul left-pix left x-ratio kern-x)
    top-mul (pixel-mul top-pix top y-ratio kern-y)]
   (api/mul
    (read-clamped-f32 input in-height in-width
                      top-pix left-pix dst-c)
    (api/mul left-mul top-mul))))


(defn- bilinear-reduction-reduce-fn
  [img-dtype driver]
  (let [in-width (api/variable "in_width")
        in-height (api/variable "in_height")
        out-width (api/variable "out_width")
        out-height (api/variable "out_height")
        n-chans (api/variable "n_channels")
        x-ratio (api/variable "x_ratio" :dtype :float32)
        y-ratio (api/variable "y_ratio" :dtype :float32)
        k-width (api/variable "k_width")
        k-height (api/variable "k_height")
        kern_x_axis (api/iteration-variable [0 k-width] "red_x" :communicative-reduce)
        kern_y_axis (api/iteration-variable [0 k-height] "red_y" :communicative-reduce)
        input (api/placeholder [in-height in-width n-chans] :dtype img-dtype)
        intermediate-op (api/compute [out-height out-width n-chans]
                                (api/tvm-fn
                                 [y x c]
                                 (api/commutative-reduce
                                  (api/tvm-fn
                                   [lhs rhs]
                                   (api/add lhs rhs))
                                  (api/const 0 :dtype :float32)
                                  :float32
                                  [(api/div (read-src-pixel
                                             input in-height in-width
                                             y x c
                                             kern_y_axis kern_x_axis
                                             y-ratio x-ratio)
                                            (api/mul x-ratio y-ratio))]
                                  [kern_y_axis kern_x_axis])))
        intermediate-output (first (api/output-tensors intermediate-op))
        int-schedule (registry/schedule-injective driver intermediate-op)
        compute-op (api/compute [out-height out-width n-chans]
                                (api/tvm-fn
                                 [y x c]
                                 (api/static-cast
                                  :uint8
                                  (clamp (api/add
                                          (float 0.5)
                                          (api/tget intermediate-output [y x c]))
                                         (float 0)
                                         (float 255)))))
        schedule (registry/schedule-injective driver compute-op)
        output (first (api/output-tensors compute-op))
        fn-name (str "bilinear_downsample_" (name img-dtype))
        lowered-fn (api/schedule->lowered-function
                    schedule [input output k-height k-width y-ratio x-ratio]
                    api/default-build-config
                    :name fn-name)
        module (registry/->module driver [lowered-fn])]
    (c/get-module-function module fn-name)))


(defn- do-bilinear-reduce-reduction
  [input output reduction-fn]
  (let [[in-height in-width in-chan] (ct/shape input)
        [out-height out-width out-chan] (ct/shape output)
        img-dtype (dtype/get-datatype input)
        filter-height (/ (double in-height)
                         (double out-height))
        filter-width (/ (double in-width)
                        (double out-width))
        kernel-height (bilinear-filter-pixel-size in-height out-height)
        kernel-width (bilinear-filter-pixel-size in-width out-width)]
    (c/call-function reduction-fn input output
                     kernel-height kernel-width
                     filter-height filter-width)
    output))


(defn test-bilinear-reduction-reduce
  []
  (first
   (verify-tensor/tensor-context
    (registry/get-driver :cpu)
    :uint8
    (let [input (ct/->tensor (->> (range (* 4 4))
                                  (partition 1)
                                  (partition 4)))
          output (ct/new-tensor [3 3 1]
                                :datatype :uint8
                                :init-value 0)
          red-fn (bilinear-reduction-reduce-fn :uint8 (drv/get-driver
                                                       drv/*current-compute-device*))]
      (do-bilinear-reduction input output red-fn)
      {:dst
       (m/array (->> (ct/to-float-array output)
                     (partition 2)))
       :src (->> (range (* 4 4))
                 (partition 4))}))))


(defn test-convolve-filter
  []
  (first
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
          input-seq (->> (range (* 4 4 1))
                         (partition 1)
                         (partition 4))
          src-tensor (ct/->tensor input-seq)
          dst-tensor (ct/new-tensor [2 2 1] :datatype :float32 :init-value 0)]

      (c/call-function actual-fn src-tensor dst-tensor 2 1)
      {:src-ary (m/array input-seq)
       :dst-ary (m/array (->> (ct/to-float-array dst-tensor)
                              (partition 1)
                              (partition 2)))}))))
