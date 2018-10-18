(ns tech.compute.tvm.image.resize
  (:require [tech.compute.tensor :as ct]
            [tech.compute.driver :as drv]
            [tech.compute.tvm.cpu]
            [tech.compute.tvm.tensor-math]
            [clojure.core.matrix :as m]
            [tech.datatype.base :as dtype]
            [tvm-clj.api :as api]
            [tech.compute.verify.tensor :as verify-tensor]
            ;;Add in syntactic sugar
            [tvm-clj.operations :refer :all]
            ))


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
  (-> (min value val_max)
      (max val_min)))


(defn- read-clamped-f32
  [img in-height in-width y x c]
  (-> (img [(clamp y 0 (- in-height 1))
            (clamp x 0 (- in-width 1))
            c])
      (cast :float32)))


(defn- area-filter-pixel-size
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

(defn- area-addr-mul
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
  (tif (= (int 0) item-idx)
       (clamp (- (+ (float 1) start-pix)
                 start)
              (float 0.0)
              (float 1.0))
       (clamp (- (+ start item-range)
                 start-pix)
              (float 0.0)
              (float 1.0))))


(defn create-kernel-op
  [out-cols k-size ratio k-name]
  (compute
   [out-cols k-size]
   (lambda
    [out-col-idx k-idx]
    (tlet
     [start (* ratio out-col-idx)
      start-pix (floor (+ start k-idx))]
     (/ (pixel-mul start-pix start ratio k-idx)
        ratio)))
   k-name))


(defn final-cast-fn
  [img-dtype input fn-name]
  (let [[in-rows in-cols in-chan] (:shape input)]
    (compute
     [in-rows in-cols in-chan]
     (lambda
      [y x c]
      (-> (input [y x c])
          (+ (float 0.5))
          (cast img-dtype)))
     fn-name)))


(defn input-coord
  [dest-coord ratio kernel-idx]
  (-> (* dest-coord ratio)
      (cast :int32)
      (+ kernel-idx)))


(defn area-reduction-fn
  "Instead of computing the kernels inline we abstract them into vectors"
  [img-dtype]
  (let [in-width (tvar "in_width")
        in-height (tvar "in_height")
        out-width (tvar "out_width")
        out-height (tvar "out_height")
        n-chans (tvar "n_channels")
        x-ratio (tvar "x_ratio" :dtype :float32)
        y-ratio (tvar "y_ratio" :dtype :float32)
        k-width (tvar "k_width")
        k-height (tvar "k_height")
        kern-x-vec (create-kernel-op out-width k-width x-ratio "kernel-x-op")
        kern-y-vec (create-kernel-op out-height k-height y-ratio "kernel-y-op")
        kern-x-axis (api/iteration-variable [0 k-width] "red_x" :communicative-reduce)
        kern-y-axis (api/iteration-variable [0 k-height] "red_y" :communicative-reduce)
        input (placeholder [in-height in-width n-chans] "input" :dtype img-dtype)
        intermediate-output (compute
                             [out-height out-width n-chans]
                             (lambda
                              [y x c]
                              (api/commutative-reduce
                               (lambda
                                [lhs rhs]
                                (+ lhs rhs))
                               (float 0)
                               :float32
                               [(* (read-clamped-f32
                                    input in-height in-width
                                    (input-coord y y-ratio kern-y-axis)
                                    (input-coord x x-ratio kern-x-axis)
                                    c)
                                   (* (kern-x-vec [x kern-x-axis])
                                      (kern-y-vec [y kern-y-axis])))]
                               [kern-y-axis kern-x-axis]))
                             "area_reduction")
        output (final-cast-fn img-dtype intermediate-output "area_cast")]
    {:input input
     :output output
     :kern-width k-width
     :kern-height k-height
     :x-ratio x-ratio
     :y-ratio y-ratio
     :reduce-op (:op intermediate-output)
     :final-op (:op output)
     :kern-x-op (:op kern-x-vec)
     :kern-y-op (:op kern-y-vec)}))


(defn schedule-area-reduction
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
        (area-reduction-fn img-dtype)

        arglist [input output
                 kern-width x-ratio
                 kern-height y-ratio]
        fn-name "area_reduce"
        schedule (api/create-schedule [final-op])
        kern-x-stage (schedule kern-x-op)
        kern-y-stage (schedule kern-y-op)
        reduce-stage (schedule reduce-op)
        final-op-stage (schedule final-op)
        intermediate-axis (:axis reduce-op)
        [int-y-axis int-x-axis int-channels] intermediate-axis
        reduce-result (first (api/output-tensors reduce-op))]
    (if (= device-type :cpu)
      (let [[final-y final-x final-chan] (:axis final-op)
            [y-outer x-outer y-inner x-inner] (api/stage-tile final-op-stage
                                                              final-y
                                                              final-x
                                                              16, 16)]
        (api/stage-compute-at reduce-stage final-op-stage final-chan)
        (api/stage-compute-at (schedule kern-x-op) final-op-stage x-inner)
        (api/stage-compute-at kern-y-stage final-op-stage y-inner)
        (api/stage-parallel final-op-stage y-inner))

      ;;Each gpu block gets a 16x16 grid
      ;;each gpu thread gets 1 pixel
      ;;This allows the reduction summation to be simple *and* gives the
      ;;caching mechanism of the GPU a chance.
      (let [[final-y final-x final-chan] (:axis final-op)
            x-chan-fused (api/stage-fuse final-op-stage [final-x final-chan])
            [y-outer x-outer y-inner x-inner] (api/stage-tile final-op-stage
                                                              final-y
                                                              x-chan-fused
                                                              16, 16)
            reduce-block-axis (api/stage-fuse final-op-stage [y-outer x-outer])
            reduce-thread-axis (api/stage-fuse final-op-stage [y-inner x-inner])]
        (api/stage-compute-at reduce-stage final-op-stage reduce-thread-axis)
        (api/stage-bind-gpu final-op-stage [reduce-block-axis] [reduce-thread-axis])
        (api/stage-gpu-injective kern-x-stage kern-x-op)
        (api/stage-gpu-injective kern-y-stage kern-y-op)))
    (if print-schedule?
      (api/schedule->str schedule arglist fn-name)
      (let [module-data (api/schedules->fns [{:schedule schedule
                                              :name :area-reduce
                                              :arglist arglist}]
                                            :target-name device-type)
            area-fn (get-in module-data [:fn-map :area-reduce])]
        area-fn))))


(defn area-reduction!
  [input output area-fn]
  (let [[in-height in-width in-chan] (ct/shape input)
        [out-height out-width out-chan] (ct/shape output)
        filter-height (/ (double in-height)
                         (double out-height))
        filter-width (/ (double in-width)
                        (double out-width))
        kernel-height (area-filter-pixel-size in-height out-height)
        kernel-width (area-filter-pixel-size in-width out-width)]
    (area-fn input output
                 kernel-width filter-width
                 kernel-height filter-height)
    output))




(defn bilinear-filter-start
  [dest-idx ratio]
  (max 0
       (-
        (* (+ 0.5 dest-idx) (double ratio))
        1)))


(defn bilinear-start-factor
  [bilinear-start]
  (- (Math/ceil bilinear-start)
     bilinear-start))


(defn create-filter-kernel-op
  [out-cols ratio k-name]
  (api/compute
   [out-cols (api/const (int 2))]
   (api/tvm-fn
    [out-col-idx k-idx]
    (api/tvm-let
     [start (api/mul ratio (api/add (api/const (float 0.5))
                                    out-col-idx))
      start-factor (api/sub (api/ceil start)
                            start)]
     (api/select (api/eq k-idx 0)
                 start-factor
                 (api/sub (api/const (float 1.0)) start-factor))))
   k-name))


(defn bilinear-filter-fn
  "Classic bilinear reduction"
  [img-dtype]
  (let [in-width (tvar "in_width")
        in-height (tvar "in_height")
        out-width (tvar "out_width")
        out-height (tvar "out_height")
        n-chans (tvar "n_channels")
        x-ratio (tvar "x_ratio" :dtype :float32)
        y-ratio (tvar "y_ratio" :dtype :float32)
        k-width (api/const (int 2))
        k-height (api/const (int 2))
        kern-x-op (create-filter-kernel-op out-width x-ratio "kernel-x-op")
        kern-y-op (create-filter-kernel-op out-height y-ratio "kernel-y-op")
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
                              (input-coord (api/add y (float 0.5)) y-ratio kern_y_axis)
                              (input-coord (api/add x (float 0.5)) x-ratio kern_x_axis)
                              c)
                             (api/mul
                              (api/tget kern-x-vec [x kern_x_axis])
                              (api/tget kern-y-vec [y kern_y_axis])))]
                           [kern_y_axis kern_x_axis]))
                         "bilinear_filter")
        intermediate-output (first (api/output-tensors intermediate-op))
        output (final-cast-fn img-dtype intermediate-output "bilinear_cast")]
    {:input input
     :output output
     :x-ratio x-ratio
     :y-ratio y-ratio
     :reduce-op intermediate-op
     :final-op (:op output)
     :kern-x-op kern-x-op
     :kern-y-op kern-y-op}))


(defn schedule-bilinear-filter-fn
  [& {:keys [device-type
             img-dtype
             print-schedule?]
      :or {device-type :cpu
           img-dtype :uint8
           print-schedule? false}}]
  (let [{:keys [input
                output
                x-ratio
                y-ratio
                reduce-op
                final-op
                kern-x-op
                kern-y-op]}
        (bilinear-filter-fn img-dtype)

        arglist [input output x-ratio y-ratio]
        fn-name :bilinear-filter
        schedule (api/create-schedule [final-op])
        stage-map (get schedule :stage_map)
        kern-x-stage (schedule kern-x-op)
        kern-y-stage (schedule kern-y-op)
        reduce-stage (schedule reduce-op)
        final-op-stage (schedule final-op)
        intermediate-axis (:axis reduce-op)
        [int-y-axis int-x-axis int-channels] intermediate-axis
        reduce-result (first (api/output-tensors reduce-op))]
    (if (= device-type :cpu)
      (let [[final-y final-x final-chan] (:axis final-op)
            [y-outer x-outer y-inner x-inner] (api/stage-tile final-op-stage
                                                              final-y
                                                              final-x
                                                              16, 16)]
        (api/stage-compute-at reduce-stage final-op-stage final-chan)
        (api/stage-compute-at kern-x-stage final-op-stage x-inner)
        (api/stage-compute-at kern-y-stage final-op-stage y-inner)
        (api/stage-parallel final-op-stage y-inner))

      ;;Each gpu block gets a 16x16 grid
      ;;each gpu thread gets 1 pixel
      ;;This allows the reduction summation to be simple *and* gives the
      ;;caching mechanism of the GPU a chance.
      (let [[final-y final-x final-chan] (:axis final-op)
            x-chan-fused (api/stage-fuse final-op-stage [final-x final-chan])
            [y-outer x-outer y-inner x-inner] (api/stage-tile final-op-stage
                                                              final-y
                                                              x-chan-fused
                                                              16, 16)
            reduce-block-axis (api/stage-fuse final-op-stage [y-outer x-outer])
            reduce-thread-axis (api/stage-fuse final-op-stage [y-inner x-inner])]
        (api/stage-compute-at reduce-stage final-op-stage reduce-thread-axis)
        (api/stage-bind-gpu final-op-stage [reduce-block-axis] [reduce-thread-axis])
        (api/stage-gpu-injective kern-x-stage kern-x-op)
        (api/stage-gpu-injective kern-y-stage kern-y-op)))
    (if print-schedule?
      (api/schedule->str schedule arglist (name fn-name))
      (let [module-data (api/schedules->fns [{:schedule schedule
                                              :name fn-name
                                              :arglist arglist}]
                                            :target-name device-type)
            bilinear-fn (get-in module-data [:fn-map fn-name])]
        bilinear-fn))))


(defn bilinear-filter!
  [input output filter-fn]
  (let [[in-height in-width in-chan] (ct/shape input)
        [out-height out-width out-chan] (ct/shape output)
        filter-height (/ (double in-height)
                         (double out-height))
        filter-width (/ (double in-width)
                        (double out-width))
        kernel-height (area-filter-pixel-size in-height out-height)
        kernel-width (area-filter-pixel-size in-width out-width)]
    (filter-fn input output filter-width filter-height)
    output))
