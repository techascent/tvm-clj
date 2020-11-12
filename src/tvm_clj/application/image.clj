(ns tvm-clj.application.image
  "Image resize algorithms showing somewhat nontrivial application
  of TVM operators."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.tensor.dimensions :as dims]
            [tech.v3.libs.buffered-image :as bufimg]
            [tvm-clj.ast :as ast]
            [tvm-clj.ast.elemwise-op :as ast-op]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tvm-clj.module :as module]
            [tvm-clj.device :as device]
            [tech.v3.resource :as resource])
  (:import [tech.v3.datatype NDBuffer ObjectReader]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn- n-img-shape
  "Normalize shape to dimensions."
  [shape-vec]
  (case (count shape-vec)
    2 (vec (concat [1] shape-vec [1]))
    3 (vec (concat [1] shape-vec))
    4 shape-vec))


(defn- to-long-round
  ^long [value]
  (long (Math/round (double value))))


(defn- to-long-ceil
  ^long [value]
  (long (Math/ceil value)))


(defn- area-filter-pixel-size
  "The size of the destination pixel in input pixels on one axis."
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


(defn area-reduction!
  [input output area-fn]
  (let [[in-height in-width in-chan] (dtype/shape input)
        [out-height out-width out-chan] (dtype/shape output)
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


(defn- clamp
  ^double [^double value ^double val_min ^double val_max]
  (-> (min value val_max)
      (max val_min)))


(defn- clamp-long
  ^long [^long value ^long val_min ^long val_max]
  (-> (min value val_max)
      (max val_min)))


(defn- compute-tensor
  [output-shape per-pixel-op datatype]
  (let [dims (dims/dimensions output-shape)
        output-shape (long-array output-shape)
        n-dims (count output-shape)
        n-elems (long (apply * output-shape))
        shape-chan (aget output-shape (dec n-dims))
        shape-x (when (> n-dims 1)
                  (aget output-shape (dec (dec n-dims))))]
    (dtt/construct-tensor
     (reify ObjectReader
       (elemwiseDatatype [rdr] datatype)
       (lsize [rdr] n-elems)
       (readObject [rdr idx]
         (case n-dims
           1 (per-pixel-op idx)
           2 (per-pixel-op (quot idx shape-chan)
                           (rem idx shape-chan))
           3 (let [c (rem idx shape-chan)
                   xy (quot idx shape-chan)
                   x (rem xy (long shape-x))
                   y (quot xy (long shape-x))]
               (per-pixel-op y x c))
           (let [local-data (long-array n-dims)]
             (loop [fwd-dim-idx 0
                    idx idx]
               (when (and (> idx 0) (< fwd-dim-idx n-dims))
                 (let [dim-idx (- n-dims fwd-dim-idx 1)
                       local-shape (aget output-shape dim-idx)
                       cur (rem idx local-shape)]
                   (aset local-data dim-idx cur)
                   (recur (unchecked-inc fwd-dim-idx)
                          (quot idx (aget output-shape dim-idx))))))
             (apply per-pixel-op local-data)))))
     dims)))


(defn- src-coord
  ^long [^long dest-coord ^long kernel-idx ^long kernel-width ^double out-over-in]
  (- (+ (Math/round (/ dest-coord out-over-in))
        kernel-idx)
     (quot kernel-width 2)))


(defn jvm-area-resize-algo
  [input output-shape]
  (let [[^long in-height ^long in-width n-chan] (dtype/shape input)
        [^long out-height ^long out-width n-chan] output-shape
        input (dtt/ensure-tensor input)
        max-idx-x (dec in-width)
        max-idx-y (dec in-height)
        x-ratio (double (/ (double out-width) in-width))
        y-ratio (double (/ (double out-height) in-height))
        x-kernel-width (/ 1.0 x-ratio)
        y-kernel-width (/ 1.0 y-ratio)
        divisor (* x-ratio y-ratio)
        reducer (fn [^double accum ^double input]
                  (+ accum input))
        identity-value 0.0]
    (compute-tensor
     [out-height out-width n-chan]
     (fn [^long y ^long x ^long c]
       (-> (loop [k-idx-y 0
                  outer-sum identity-value]
             (if (< k-idx-y y-kernel-width)
               (recur (unchecked-inc k-idx-y)
                      (double
                       (loop [k-idx-x 0
                              inner-sum outer-sum]
                         (if (< k-idx-x x-kernel-width)
                           (let [src-coord-x (clamp-long
                                              (src-coord x k-idx-x x-kernel-width x-ratio)
                                              0
                                              max-idx-x)
                                 src-coord-y (clamp-long
                                              (src-coord y k-idx-y y-kernel-width y-ratio)
                                              0
                                              max-idx-y)]
                             (recur (unchecked-inc k-idx-x)
                                    (double
                                     (reducer inner-sum (.ndReadDouble input src-coord-y
                                                                       src-coord-x c)))))
                           inner-sum))))
               outer-sum))
           (double)
           (* divisor)
           (clamp 0.0 255.0)
           (unchecked-long)))
     :uint8)))


(defn jvm-area-resize-fn!
  [input output]
  (dtype/copy! (jvm-area-resize-algo input (dtype/shape output))
               output)
  output)


(defn tvm-area-resize-algo-def
  []
  (let [n-chan (ast/variable "n-chan")
        in-width (ast/variable "in-width")
        in-height (ast/variable "in-height")
        out-width (ast/variable "out-width")
        out-height (ast/variable "out-height")
        input (ast/placeholder [in-height in-width n-chan] "input" :dtype :uint8)
        max-idx-x (ast-op/- in-width (int 1))
        max-idx-y (ast-op/- in-height (int 1))
        x-ratio (ast-op// (ast-op/cast out-width :float32)
                          (ast-op/cast in-width :float32))
        y-ratio (ast-op// (ast-op/cast out-height :float32)
                          (ast-op/cast in-height :float32))
        x-kernel-width (ast-op// (float 1.0) x-ratio)
        y-kernel-width (ast-op// (float 1.0) y-ratio)
        divisor (ast-op/* x-ratio y-ratio)
        clamp-fn (fn [val val-min val-max]
                   (-> (ast-op/min val val-max)
                       (ast-op/max val-min)))
        coord-fn (fn [dest-coord kernel-idx kernel-width out-over-in]
                   (-> (ast-op// (ast-op/cast dest-coord :float32) out-over-in)
                       (ast-op/+ (ast-op/cast kernel-idx :float32))
                       (ast-op/- (ast-op// kernel-width (float 2.0)))
                       (ast-op/cast :int32)))
        compute-op (ast/compute
                    [out-height out-width n-chan]
                    (ast/tvm-fn
                     [y x c]
                     (ast/commutative-reduce
                      (ast/tvm-fn->commutative-reducer
                       (ast/tvm-fn [lhs rhs] (ast-op/+ lhs rhs))
                       [(float 0.0)])
                      [{:domain [0 y-kernel-width]
                        :name "k-idx-y"}
                       {:domain [0 x-kernel-width]
                        :name "k-idx-x"}]
                      [(fn [k-idx-y k-idx-x]
                         (-> (ast/tget input
                                       [(-> (coord-fn y k-idx-y y-kernel-width y-ratio)
                                            (clamp-fn (int 0) max-idx-y))
                                        (-> (coord-fn x k-idx-x x-kernel-width x-ratio)
                                            (clamp-fn (int 0) max-idx-x))
                                        c])
                             (ast-op/cast :float32)))]))
                    "partial_result")
        ;;Result in floating point space.
        partial-result (first (ast/output-tensors compute-op))
        result-op (ast/compute
                   [out-height out-width n-chan]
                   (ast/tvm-fn
                    [y x c]
                    (-> (ast/tget partial-result [y x c])
                        (ast-op/* divisor)
                        (clamp-fn (float 0) (float 255))
                        (ast-op/cast :uint8)))
                   "result")
        output (first (ast/output-tensors result-op))]
    {:arguments [input output]
     :reduce-kernel compute-op
     :final-kernel result-op}))


(defn schedule-tvm-area
  [{:keys [arguments reduce-kernel final-kernel]}]
  (let [schedule (schedule/create-schedule final-kernel)
        stage-map (:stage_map schedule)
        reduce-stage (get stage-map reduce-kernel)
        final-stage (get stage-map final-kernel)
        [out-y out-x out-chan] (:axis final-kernel)
        final-axis (schedule/stage-fuse final-stage (:axis final-kernel))]
    (schedule/stage-compute-at reduce-stage final-stage final-axis)
    (schedule/stage-parallel final-stage final-axis)
    {:arguments arguments
     :schedule schedule}))


(defn compile-scheduled-tvm-area
  [scheduled]
  (let [module (compiler/compile {"cpu_area" scheduled})
        low-level-fn (module/find-function module "cpu_area")
        ref-map {:module module}]
    (fn [input output]
      (let [tvm-input (dtt/clone input :container-type :native-heap)
            tvm-output (device/device-tensor output :cpu 0)]
        ;;;Dereference ref-map
        (ref-map :module)
        (low-level-fn tvm-input tvm-output)
        (dtype/copy! tvm-output output)))))


(defn area-resize!
  [input ^long new-width resize-fn]
  (let [[^long height ^long width _nchan] (dtype/shape input)
        ratio (double (/ new-width width))
        new-height (Math/round (* height ratio))
        output-img (bufimg/new-image new-height new-width
                                     (bufimg/image-type input))]
    (resize-fn (dtt/ensure-tensor input) (dtt/ensure-tensor output-img))
    output-img))


(comment
  (def input-img (bufimg/load "test/data/jen.jpg"))
  (def test-fn (-> (tvm-area-resize-algo-def)
                   (schedule-tvm-area)
                   (compile-scheduled-tvm-area)))

  (def result (time (area-resize! input-img 512 test-fn)))
  ;;179 ms
  (def jvm-result (time (area-resize! input-img 512 jvm-area-resize-fn!)))
  ;;5.7 seconds

  )
