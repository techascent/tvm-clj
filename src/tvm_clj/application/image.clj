(ns tvm-clj.application.image
  "Image resize algorithm showing somewhat nontrivial application
  of TVM operators.  In this case we have an algorithm which is a simple
  average area color algorithm used for scaling images down.  This reads a
  rectangle in the source image and averages it for every destination pixel.

  This is a namespace where you want to view the source :-)

```clojure
  (def input-img (bufimg/load \"test/data/jen.jpg\"))
  (def test-fn (-> (tvm-area-resize-algo-def)
                   (schedule-tvm-area)
                   (compile-scheduled-tvm-area)))

  (def result (time (area-resize! input-img 512 test-fn)))
  ;;179 ms
  (def jvm-result (time (area-resize! input-img 512 jvm-area-resize-fn!)))
  ;;5.7 seconds
```"
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.tensor.dimensions :as dims]
            [tech.v3.libs.buffered-image :as bufimg]
            [tvm-clj.ast :as ast]
            [tvm-clj.ast.elemwise-op :as ast-op]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tvm-clj.module :as module]
            [tvm-clj.device :as device])
  (:import [tech.v3.datatype NDBuffer ObjectReader]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn- clamp
  ^double [^double value ^double val_min ^double val_max]
  (-> (min value val_max)
      (max val_min)))


(defn- clamp-long
  ^long [^long value ^long val_min ^long val_max]
  (-> (min value val_max)
      (max val_min)))


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
    (dtt/compute-tensor
     [out-height out-width n-chan]
     ;;Micro optimization in order to avoid boxing y,x,c on every index
     ;;access.  Yes, this does have a noticeable perf impact :-)
     (reify NDBuffer
       (ndReadObject [this y x c]
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
             (unchecked-long))))
     :uint8)))


(defn jvm-area-resize-fn!
  [input output]
  (dtype/copy! (jvm-area-resize-algo input (dtype/shape output))
               output)
  output)


(defn tvm-area-resize-algo-def
  "Step 1 is to define the algorithm.  This definition looks strikingly similar
  to the definition above."
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

                      ;;First arg is a commutative reducer.
                      (ast/tvm-fn->commutative-reducer
                       ;;Here is our reducing function.
                       (ast/tvm-fn [lhs rhs] (ast-op/+ lhs rhs))
                       ;;Zero is the identity operation for this reduction.
                       [(float 0.0)])

                      ;;Next are the inner axis we will reduce over
                      [{:domain [0 y-kernel-width]
                        :name "k-idx-y"}
                       {:domain [0 x-kernel-width]
                        :name "k-idx-x"}]

                      ;;Finally a function from reduction axes to every input
                      ;;argument as defined by our reducer above.
                      [(fn [k-idx-y k-idx-x]
                         (-> (ast/tget input
                                       [(-> (coord-fn y k-idx-y y-kernel-width y-ratio)
                                            (clamp-fn (int 0) max-idx-y))
                                        (-> (coord-fn x k-idx-x x-kernel-width x-ratio)
                                            (clamp-fn (int 0) max-idx-x))
                                        c])
                             (ast-op/cast :float32)))]))
                    ;;Finally the name so if we want to see the intermediate represetation we can
                    ;;tell what it is.
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
  "Step 2 is to 'schedule' the algorithm, thus mapping it to a particular
  hardware backend and definining where parallelism is safe."
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
  "Step 3 you compile it to a module, find the desired function, and
  wrap it with whatever wrapping code you need."
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
  "Perform an area resize with a defined resize algorith."
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
