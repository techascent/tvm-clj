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

(defn- gpu-injective
  [stage op & {:keys [thread-count]
               :or {thread-count 32}}]
  (let [fused-axis (api/stage-fuse stage (:axis op))
        [bx tx] (api/split-stage-by-factor stage fused-axis thread-count)]
    (api/bind-gpu-axis stage [bx] [tx])))

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
        stage-map (get schedule :stage_map)
        kern-x-stage (get stage-map kern-x-op)
        kern-y-stage (get stage-map kern-y-op)
        reduce-stage (get stage-map reduce-op)
        final-op-stage (get stage-map final-op)
        intermediate-axis (:axis reduce-op)
        [int-y-axis int-x-axis int-channels] intermediate-axis
        reduce-result (first (api/output-tensors reduce-op))]
    (if (= device-type :cpu)
      ;;Caching shown for demonstration.  Made no difference on test machine.
      (let [{cached-output :tensor
             schedule :schedule
             cache-write-op :tensor-op} (api/schedule-cache-write
                                         schedule
                                         reduce-result "local")
            stage-map (:stage_map schedule)
            reduce-stage (get stage-map reduce-op)
            kern-x-stage (get stage-map kern-x-op)
            kern-y-stage (get stage-map kern-y-op)
            cache-write-stage (get stage-map cache-write-op)
            [y-outer x-outer y-inner x-inner] (api/stage-tile reduce-stage
                                                              int-y-axis
                                                              int-x-axis
                                                              64 64)]
        ;;CPU reduction is simple really.  Tile the result and parallelize
        ;;over the image with large enough tiles that each thread gets some
        ;;significant work before a context switch but without thrashing l1.
        (api/stage-compute-at kern-y-stage reduce-stage y-outer)
        (api/stage-compute-at kern-x-stage reduce-stage x-outer)
        ;; (api/stage-compute-at kern-x-stage reduce-stage x-outer)
        (api/stage-compute-at cache-write-stage reduce-stage x-outer)
        (api/stage-parallel reduce-stage x-outer)
        (api/stage-parallel final-op-stage (api/stage-fuse
                                            final-op-stage
                                            (:axis final-op))))

      ;;Each gpu block gets a 16x16 grid
      ;;each gpu thread gets 1 pixel
      ;;This allows the reduction summation to be simple *and* gives the
      ;;caching mechanism of the GPU a chance.
      (let [x-chan-fused (api/stage-fuse reduce-stage [int-x-axis int-channels])
            [y-outer x-outer y-inner x-inner] (api/stage-tile reduce-stage
                                                              int-y-axis
                                                              x-chan-fused
                                                              8 8)
            reduce-block-axis (api/stage-fuse reduce-stage [y-outer x-outer])
            reduce-thread-axis (api/stage-fuse reduce-stage [y-inner x-inner])]
        (api/bind-gpu-axis reduce-stage [reduce-block-axis] [reduce-thread-axis])
        (gpu-injective final-op-stage final-op)
        (gpu-injective kern-x-stage kern-x-op)
        (gpu-injective kern-y-stage kern-y-op)))
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
