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
            [tvm-clj.device :as device]
            [primitive-math :as pmath])
  (:import [tech.v3.datatype NDBuffer ObjectReader]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn area-resize!
  "Perform an area resize with a defined resize algorithm."
  [input ^long new-width resize-fn]
  (let [[^long height ^long width _nchan] (dtype/shape input)
        ratio (double (/ new-width width))
        new-height (Math/round (* height ratio))
        output-img (bufimg/new-image new-height new-width
                                     (bufimg/image-type input))]
    (resize-fn (dtt/ensure-tensor input) (dtt/ensure-tensor output-img))
    output-img))


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
        x-ratio (/ (double out-width) (double in-width))
        y-ratio (/ (double out-height) (double in-height))
        ;;Size of the reduction rectangle in the X dimension
        ;;Size of the reduction rectangle in the Y dimension
        reduce-kernel-width (/ 1.0 x-ratio)
        reduce-kernel-height (/ 1.0 y-ratio)
        divisor (* x-ratio y-ratio)
        identity-value 0.0]
    ;;Define a tensor using an algorithm definition - a 'compute' tensor
    (dtt/typed-compute-tensor
     ;;Datatype for result
     :uint8
     ;;Output shape
     [out-height out-width n-chan]
     ;;Argument names used in code block below
     [y x c]
     ;;Tensor definition.  This is compiled inline into a new reified tensor
     ;;type.
     (-> (loop [k-idx-y 0
                outer-sum identity-value]
           (if (< k-idx-y reduce-kernel-height)
             (recur (unchecked-inc k-idx-y)
                    (double
                     (loop [k-idx-x 0
                            inner-sum outer-sum]
                       (if (< k-idx-x reduce-kernel-width)
                         (let [src-coord-x (clamp-long
                                            (src-coord x k-idx-x reduce-kernel-width x-ratio)
                                            0
                                            max-idx-x)
                               src-coord-y (clamp-long
                                            (src-coord y k-idx-y reduce-kernel-height y-ratio)
                                            0
                                            max-idx-y)]
                           (recur (unchecked-inc k-idx-x)
                                  (pmath/+ inner-sum (.ndReadDouble input src-coord-y
                                                                    src-coord-x c))))
                         inner-sum))))
             outer-sum))
         (double)
         (* divisor)
         (clamp 0.0 255.0)
         (unchecked-long)))))


(defn jvm-area-split-resize-algo
  [input output-shape]
  (let [[^long in-height ^long in-width n-chan] (dtype/shape input)
        [^long out-height ^long out-width n-chan] output-shape
        input (dtt/ensure-tensor input)
        max-idx-x (dec in-width)
        max-idx-y (dec in-height)
        x-ratio (double (/ (double out-width) in-width))
        y-ratio (double (/ (double out-height) in-height))
        reduce-kernel-width (/ 1.0 x-ratio)
        reduce-kernel-height (/ 1.0 y-ratio)
        divisor (* x-ratio y-ratio)
        identity-value 0.0

        ;;If we split the X,y calculations we can be more work and
        ;;memory efficient as we do not recalculate as many partial
        ;;summations
        horiz-sum (dtt/typed-compute-tensor
                   ;;datatype
                   :float64
                   ;;shape
                   [in-height out-width n-chan]
                   ;argument names
                   [y x c]
                   ;;per-element read code block
                   (loop [k-idx-x 0
                          inner-sum identity-value]
                     (if (< k-idx-x reduce-kernel-width)
                       (let [src-coord-x (clamp-long
                                          (src-coord x k-idx-x reduce-kernel-width x-ratio)
                                          0
                                          max-idx-x)
                             src-coord-y y]
                         (recur (unchecked-inc k-idx-x)
                                (pmath/+ inner-sum (.ndReadDouble input src-coord-y
                                                                  src-coord-x c))))
                       inner-sum)))
        ;;Force the calculation to complete to a temporary
        temp-img (dtt/ensure-tensor (dtype/clone horiz-sum))]
    ;;Return the defined but not executed result.
    (dtt/typed-compute-tensor
     ;;datatype
     :uint8
     ;;shape
     [out-height out-width n-chan]
     ;;per element arg names
     [y x c]
     ;;code block
     (-> (loop [k-idx-y 0
                outer-sum identity-value]
           (if (< k-idx-y reduce-kernel-height)
             (let [src-coord-x x
                   src-coord-y (clamp-long
                                (src-coord y k-idx-y reduce-kernel-height y-ratio)
                                0
                                max-idx-y)]
               (recur (unchecked-inc k-idx-y)
                      (pmath/+ outer-sum (.ndReadDouble temp-img src-coord-y
                                                        src-coord-x c))))
             outer-sum))
         (double)
         (* divisor)
         (clamp 0.0 255.0)
         (unchecked-long)))))


(defn jvm-area-resize-fn!
  [jvm-resize-algo input output]
  (dtype/copy! (jvm-resize-algo input (dtype/shape output))
               output)
  output)


(defn compile-cpu-scheduled-tvm-area
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


(defn compile-gpu-scheduled-tvm-area
  [gpu-schedule device-type]
  (let [module (compiler/compile
                {"gpu_area" (assoc gpu-schedule :target device-type)})
        low-level-fn (module/find-function module "gpu_area")]
    (fn [input output]
      (let [cpu-input (dtt/clone input :container-type :native-heap)
            tvm-input (device/cpu->device cpu-input device-type 0)
            tvm-output (device/device-tensor output device-type 0)
            ref-map {:module module}]
        ;;;Dereference ref-map
        (ref-map :module)
        (low-level-fn tvm-input tvm-output)
        (dtype/copy! (device/device->cpu tvm-output)
                     output)))))


(defn tvm-area-resize-algo
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
        x-ratio (ast-op// (ast-op/cast out-width :float32) (ast-op/cast in-width :float32))
        y-ratio (ast-op// (ast-op/cast out-height :float32) (ast-op/cast in-height :float32))
        ;;Size of the reduction rectangle in the X dimension
        ;;Size of the reduction rectangle in the Y dimension
        reduce-kernel-width (ast-op// (float 1.0) x-ratio)
        reduce-kernel-height (ast-op// (float 1.0) y-ratio)
        divisor (ast-op/* x-ratio y-ratio)
        clamp-fn (fn [val val-min val-max]
                   (-> (ast-op/min val val-max)
                       (ast-op/max val-min)))
        coord-fn (fn [dest-coord kernel-idx kernel-width out-over-in]
                   (-> (ast-op// (ast-op/cast dest-coord :float32) out-over-in)
                       (ast-op/+ (ast-op/cast kernel-idx :float32))
                       (ast-op/- (ast-op// kernel-width (float 2.0)))
                       (ast-op/cast :int32)))
        partial-result-op (ast/compute
                           [out-height out-width n-chan]
                           (ast/tvm-fn
                            [y x c]
                            (ast/commutative-reduce

                             ;;First arg is a commutative reducer.
                             (ast/tvm-fn->commutative-reducer
                              ;;Here is our reducing function.
                              (ast/tvm-fn [lhs rhs] (ast-op/+ lhs rhs))
                              ;;Zero is the identity operation for this reduction.
                              ;;One identity value means one accumulator variable (lhs)
                              ;;and the rest of the arguments (rhs) are interpreted as
                              ;;input arguments
                              [(float 0.0)])

                             ;;Next are the inner axis we will reduce over
                             [{:domain [0 reduce-kernel-height] :name "k-idx-y"}
                              {:domain [0 reduce-kernel-width] :name "k-idx-x"}]

                             ;;Finally a function from reduction axes to every input
                             ;;argument as defined by our reducer above.
                             [(fn [k-idx-y k-idx-x]
                                (-> (ast/tget input
                                              [(-> (coord-fn y k-idx-y reduce-kernel-height y-ratio)
                                                   (clamp-fn (int 0) max-idx-y))
                                               (-> (coord-fn x k-idx-x reduce-kernel-width x-ratio)
                                                   (clamp-fn (int 0) max-idx-x))
                                               c])
                                    ;;perform operation in float32 space
                                    (ast-op/cast :float32)))]))
                           ;;Finally the name so if we want to see the intermediate representation we can
                           ;;tell what it is.
                           "partial_result")
        ;;Result in floating point space.
        partial-result (first (ast/output-tensors partial-result-op))
        result-op (ast/compute
                   [out-height out-width n-chan]
                   (ast/tvm-fn
                    [y x c]
                    (-> (ast/tget partial-result [y x c])
                        (ast-op/* divisor)
                        (clamp-fn (float 0) (float 255))
                        ;;convert back to uint8 space
                        (ast-op/cast :uint8)))
                   "result")
        output (first (ast/output-tensors result-op))]
    {:arguments [input output]
     :partial-result-op partial-result-op
     :final-op result-op}))


(defn schedule-cpu-tvm-area
  "Step 2 is to 'schedule' the algorithm, thus mapping it to a particular
  hardware backend and definining where parallelism is safe.  This fuses all
  three axis into one loop and runs the computation in parallel."
  [{:keys [arguments partial-result-op final-op]}]
  (let [schedule (schedule/create-schedule final-op)
        stage-map (:stage_map schedule)
        partial-stage (get stage-map partial-result-op)
        final-stage (get stage-map final-op)
        [final-y final-x final-c] (:axis final-op)
        [final-y-outer final-x-outer final-y-inner final-x-inner]
        (schedule/stage-tile final-stage
                             final-y
                             final-x
                             1, 16)]
    (schedule/stage-compute-at partial-stage final-stage final-c)
    (schedule/stage-parallel final-stage final-x-outer)
    {:arguments arguments
     :schedule schedule}))


(defn schedule-gpu-tvm-area
  "Scheduling for a GPU means a 2-level breakdown of your algorithm in terms
  of 'blocks' and 'threads'.  Blocks are multiprocessor units that share
  a block of memory called 'shared-memory'."
  [{:keys [arguments partial-result-op final-op]}]
  (let [schedule (schedule/create-schedule final-op)
        stage-map (:stage_map schedule)
        partial-stage (get stage-map partial-result-op)
        final-stage (get stage-map final-op)
        [final-y final-x final-c] (:axis final-op)
        [final-y-outer final-x-outer final-y-inner final-x-inner]
        (schedule/stage-tile final-stage
                             final-y
                             final-x
                             16, 16)
        block-axis (schedule/stage-fuse final-stage [final-y-outer final-x-outer])
        thread-axis (schedule/stage-fuse final-stage [final-y-inner final-x-inner])]
    ;;Each thread does it's own reduction
    (schedule/stage-compute-at partial-stage final-stage final-c)
    (schedule/stage-bind-gpu final-stage [block-axis] [thread-axis])
    {:schedule schedule
     :arguments arguments}))



(comment
  (def input-img (bufimg/load "test/data/jen.jpg"))

  (def jvm-result (time (area-resize! input-img 512  (partial jvm-area-resize-fn! jvm-area-resize-algo))))
  ;;2.6 seconds
  (def jvm-split-result (time (area-resize!
                               input-img 512
                               (partial jvm-area-resize-fn!
                                        jvm-area-split-resize-algo))))
  ;;2 seconds

  (def tvm-cpu-fn (-> (tvm-area-resize-algo)
                      (schedule-cpu-tvm-area)
                      (compile-cpu-scheduled-tvm-area)))

  (def tvm-cpu-result (time (area-resize! input-img 512 tvm-cpu-fn)))
  ;;110ms

  (def tvm-cuda-fn (-> (tvm-area-resize-algo)
                       (schedule-gpu-tvm-area)
                       (compile-gpu-scheduled-tvm-area :cuda)))

  (def cuda-result (time (area-resize! input-img 512 tvm-cuda-fn)))
  ;;88ms

  (def tvm-opencl-fn (-> (tvm-area-resize-algo)
                         (schedule-gpu-tvm-area)
                         (compile-gpu-scheduled-tvm-area :opencl)))


  (def opencl-result (time (area-resize! input-img 512 tvm-opencl-fn)))
  ;;77ms

  )
