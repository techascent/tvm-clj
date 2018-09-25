(ns tvm-clj.image.bilinear-reduce
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
              (clamp (api/sub (api/add (float 1) start-pix)
                              start)
                     (float 0.0)
                     (float 1.0))
              (clamp (api/sub (api/add start item-range)
                              start-pix)
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
    left-pix (api/floor (api/add left kern-x))
    top-pix (api/floor (api/add top kern-y))
    left-mul (pixel-mul left-pix left x-ratio kern-x)
    top-mul (pixel-mul top-pix top y-ratio kern-y)]
   (api/mul
    (read-clamped-f32 input in-height in-width
                      (api/static-cast :int32 top-pix)
                      (api/static-cast :int32 left-pix)
                      dst-c)
    (api/mul left-mul top-mul))))


(defn- simple-compile
  [raw-fn fn-name bind-list driver]
  (let [schedule (registry/schedule-injective driver raw-fn nil)
        lowered-fn (api/schedule->lowered-function
                    schedule bind-list
                    api/default-build-config
                    :name fn-name)
        module (registry/->module driver [lowered-fn])]
    (comment (println "module-source\n"
                      (c/get-module-source module {:format "cl"})))
    (c/get-module-function module fn-name)))


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
        int-schedule (registry/schedule-injective driver intermediate-op nil)
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
        output (first (api/output-tensors compute-op))]
    (simple-compile compute-op (str "bilinear_downsample_" (name img-dtype))
                    [input output k-height k-width y-ratio x-ratio]
                    driver)))


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
      (do-bilinear-reduce-reduction input output red-fn)
      {:dst
       (m/array (->> (ct/to-float-array output)
                     (partition 2)))
       :src (->> (range (* 4 4))
                 (partition 4))}))))


(defn create-kernel-op
  [out-cols k-size ratio k-name]
  (api/compute
   [out-cols k-size]
   (api/tvm-fn
    [out-col-idx k-idx]
    (api/tvm-let
     [start (api/mul ratio out-col-idx)
      start-pix (api/floor (api/add start k-idx))]
     (api/div (pixel-mul start-pix start ratio k-idx)
              ratio)))
   k-name))


(defn final-cast-fn
  [img-dtype input fn-name]
  (let [[in-rows in-cols in-chan] (:shape input)]
    (api/compute
     [in-rows in-cols in-chan]
     (api/tvm-fn
      [y x c]
      (api/static-cast
       img-dtype
       (api/add
        (api/tget input [y x c])
        (float 0.5))))
     fn-name)))


(defn input-coord
  [dest-coord ratio kernel-idx]
  (->> (api/mul dest-coord ratio)
       (api/static-cast :int32)
       (api/add kernel-idx)))


(defn bilinear-reduction-kernel-reduce-fn
  "Instead of computing the kernels inline we abstract them into vectors"
  [img-dtype]
  (let [in-width (api/variable "in_width")
        in-height (api/variable "in_height")
        out-width (api/variable "out_width")
        out-height (api/variable "out_height")
        n-chans (api/variable "n_channels")
        x-ratio (api/variable "x_ratio" :dtype :float32)
        y-ratio (api/variable "y_ratio" :dtype :float32)
        k-width (api/variable "k_width")
        k-height (api/variable "k_height")
        kern-x-op (create-kernel-op out-width k-width x-ratio "kernel-x-op")
        kern-y-op (create-kernel-op out-height k-height y-ratio "kernel-y-op")
        kern-x-vec (first (api/output-tensors kern-x-op))
        kern-y-vec (first (api/output-tensors kern-y-op))
        kern_x_axis (api/iteration-variable [0 k-width] "red_x" :communicative-reduce)
        kern_y_axis (api/iteration-variable [0 k-height] "red_y" :communicative-reduce)
        input (api/placeholder [in-height in-width n-chans] "input" :dtype img-dtype)
        intermediate-op (api/compute
                         [out-height out-width n-chans]
                         (api/tvm-fn
                          [y x c]
                          (api/commutative-reduce
                           (api/tvm-fn
                            [lhs rhs]
                            (api/add lhs rhs))
                           (api/const 0 :dtype :float32)
                           :float32
                           [(api/mul
                             (read-clamped-f32
                              input in-height in-width
                              (input-coord y y-ratio kern_y_axis)
                              (input-coord x x-ratio kern_x_axis)
                              c)
                             (api/mul
                              (api/tget kern-x-vec [x kern_x_axis])
                              (api/tget kern-y-vec [y kern_y_axis])))]
                           [kern_y_axis kern_x_axis]))
                         "bilinear_reduction")
        intermediate-output (first (api/output-tensors intermediate-op))
        compute-op (final-cast-fn img-dtype intermediate-output "bilinear_cast")
        output (first (api/output-tensors compute-op))]
    {:input input
     :output output
     :kern-width k-width
     :kern-height k-height
     :x-ratio x-ratio
     :y-ratio y-ratio
     :reduce-op intermediate-op
     :final-op compute-op
     :kern-x-op kern-x-op
     :kern-y-op kern-y-op}))


(defn schedule-bilinear-reduce-fn
  [& {:keys [device-type
             img-dtype
             print-schedule?]
      :or {device-type :cpu
           img-dtype :uint8
           print-schedule? false}}]
  (let [{:keys [input
                output
                kern-width
                kern-height
                x-ratio
                y-ratio
                reduce-op
                final-op
                kern-x-op
                kern-y-op]}
        (bilinear-reduction-kernel-reduce-fn img-dtype)

        arglist [input output
                 kern-width x-ratio
                 kern-height y-ratio]
        fn-name "bilinear_reduce"
        schedule (api/create-schedule [final-op])
        reduce-axis (:axis reduce-op)
        [reduce-y-axis reduce-x-axis reduce-channels] reduce-axis
        stage-map (get schedule :stage_map)
        kern-x-stage (get stage-map kern-x-op)
        kern-y-stage (get stage-map kern-y-op)
        reduce-stage (get stage-map reduce-op)
        [y-outer x-outer y-inner x-inner]
        (api/stage-tile reduce-stage reduce-y-axis reduce-x-axis 16 32)
        _ (api/stage-compute-at kern-y-stage reduce-stage y-outer)
        _ (api/stage-compute-at kern-x-stage reduce-stage x-outer)
        _ (api/stage-parallel reduce-stage x-outer)
        final-op-stage (get stage-map final-op)
        fused (api/stage-fuse final-op-stage (:axis final-op))
        _ (api/stage-parallel final-op-stage fused)

        input-seq (range (* 4 4))
        input-tens (ct/->tensor (->> input-seq
                                     (partition 1)
                                     (partition 4)))
        output-tens (ct/new-tensor [2 2 1])]
    (if print-schedule?
      (api/schedule->str schedule arglist fn-name)
      (let [module-data (api/schedules->fns [{:schedule schedule
                                              :name :bilinear-reduce
                                              :arglist arglist}]
                                            :target-name device-type)
            bilinear-fn (get-in module-data [:fn-map :bilinear-reduce])]
        bilinear-fn))))


(defn bilinear-reduce!
  [input output bilinear-fn]
  (let [[in-height in-width in-chan] (ct/shape input)
        [out-height out-width out-chan] (ct/shape output)
        filter-height (/ (double in-height)
                         (double out-height))
        filter-width (/ (double in-width)
                        (double out-width))
        kernel-height (bilinear-filter-pixel-size in-height out-height)
        kernel-width (bilinear-filter-pixel-size in-width out-width)]
    (bilinear-fn input output
                 kernel-width filter-width
                 kernel-height filter-height)
    output))

(defn seq-mean
  [item-seq]
  (long
   (Math/round
    (double
     (/ (apply + item-seq)
        (count item-seq))))))


(defn test-bilinear-reduce
  []
  (let [device-type :cpu
        img-dtype :uint8]
    (first
     (verify-tensor/tensor-context
      (registry/get-driver device-type)
      img-dtype
      (let [bilinear-fn (schedule-bilinear-reduce-fn
                         :device-type :cpu
                         :img-dtype :uint8)
            input-tens (ct/->tensor (->> (range (* 4 4))
                                         (partition 1)
                                         (partition 4)))
            output-tens (ct/new-tensor [2 2 1])]
        (bilinear-fn input-tens output-tens
                     2 (float 2)
                     2 (float 2))
        (vec (ct/to-array-of-type output-tens :int32)))))))
