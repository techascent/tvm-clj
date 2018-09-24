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
  (let [schedule (registry/schedule-injective driver raw-fn)
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



(defn linear-reduce-transpose-fn
  [img-dtype & {:keys [input]}]
  (let [[in-rows in-cols n-chans] (if input
                                    (:shape input)
                                    [(api/variable "in_height")
                                     (api/variable "in_width")
                                     (api/variable "n_channels")
                                     ])
        out-cols (api/variable "out-cols")
        input (or input (api/placeholder [in-rows in-cols n-chans] :dtype img-dtype))
        ratio (api/variable "ratio" :dtype :float32)
        k-size (api/variable "k_size")
        kernel-op (api/compute
                   [out-cols k-size]
                   (api/tvm-fn
                    [out-col-idx k-idx]
                    (api/tvm-let
                     [start (api/mul ratio out-col-idx)
                      start-pix (api/floor (api/add start k-idx))]
                     (api/div (pixel-mul start-pix start ratio k-idx)
                              ratio))))
        kernel-vec (first (api/output-tensors kernel-op))
        kern-var (api/iteration-variable [0 k-size] "k_var" :communicative-reduce)
        ->pixel (fn [out-col-idx k-idx]
                  (api/static-cast
                   :int32
                   (-> (api/mul ratio out-col-idx)
                       (api/add k-idx))))
        combine-fn (api/tvm-fn
                    [dst input]
                    (api/add dst input))]
    {:fn!
     (api/compute
      [out-cols in-rows n-chans]
      (api/tvm-fn
       [col-idx row-idx chan-idx]
       (api/commutative-reduce
        combine-fn
        (api/const 0 :dtype :float32)
        :float32
        [(api/mul (read-clamped-f32 input in-rows in-cols
                                    row-idx (->pixel col-idx kern-var)
                                    chan-idx)
                  (api/tget kernel-vec [col-idx kern-var]))]
        [kern-var])))
     :input input
     :k-size k-size
     :ratio ratio}))


(defn final-cast-fn
  [img-dtype input]
  (let [[in-rows in-cols in-chan] (:shape input)]
    (api/compute
     [in-rows in-cols in-chan]
     (api/tvm-fn
      [y x c]
      (api/static-cast
       img-dtype
       (api/add
        (api/tget input [y x c])
        (float 0.5)))))))


(defn create-linear-reduce-fn
  [img-dtype]
  (let [{stage-1-fn! :fn!
         stage-1-input :input
         stage-1-k-size :k-size
         stage-1-ratio :ratio} (linear-reduce-transpose-fn img-dtype)
        stage-output (first (api/output-tensors stage-1-fn!))

        {stage-2-fn! :fn!
         stage-2-input :input
         stage-2-k-size :k-size
         stage-2-ratio :ratio} (linear-reduce-transpose-fn :float32
                                                           :input stage-output)
        stage-2-output (first (api/output-tensors stage-2-fn!))
        final-op (final-cast-fn img-dtype stage-2-output)
        output (first (api/output-tensors final-op))
        _ (println (drv/get-driver drv/*current-compute-device*))
        item-fn (simple-compile final-op "linear_reduce_uint8"
                                [stage-1-input output
                                 stage-1-k-size
                                 stage-1-ratio
                                 stage-2-k-size
                                 stage-2-ratio]
                                (drv/get-driver drv/*current-compute-device*))]
    item-fn))


(defn linear-reduce!
  [input output red-fn]
  (let [[in-height in-width in-chan] (ct/shape input)
        [out-height out-width out-chan] (ct/shape output)
        filter-height (/ (double in-height)
                         (double out-height))
        filter-width (/ (double in-width)
                        (double out-width))
        kernel-height (bilinear-filter-pixel-size in-height out-height)
        kernel-width (bilinear-filter-pixel-size in-width out-width)]

    (c/call-function red-fn input output
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

(defn test-linear-reduce
  [& {:keys [device-type]
      :or {device-type :opencl}}]
  (let [img-dtype :uint8]
    (first
     (verify-tensor/tensor-context
      (registry/get-driver device-type)
      img-dtype
      (let [input-data (->> (range (* 4 4))
                            (partition 1)
                            (partition 4))
            input-tens (ct/->tensor input-data)
            output-tens (ct/new-tensor [2 4 1] :datatype :float32)
            {stage-1-fn! :fn!
             stage-1-input :input
             stage-1-k-size :k-size
             stage-1-ratio :ratio} (linear-reduce-transpose-fn img-dtype)
            stage-output (first (api/output-tensors stage-1-fn!))
            reduce-fn (simple-compile stage-1-fn! "stage_1"
                                      [stage-1-input stage-output
                                       stage-1-k-size stage-1-ratio]
                                      (drv/get-driver drv/*current-compute-device*))]
        (c/call-function reduce-fn input-tens output-tens 2 (float 2))
        {:result (->> (ct/to-array-of-type output-tens :float32)
                   vec)})))))
