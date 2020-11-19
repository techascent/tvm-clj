(ns tvm-clj.application.kmeans
  "TVM/JVM comparison of the kmeans algorithm components."
  (:require [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.reductions :as dtype-reduce]
            [tech.v3.datatype.nio-buffer :as nio-buffer]
            [tech.v3.tensor :as dtt]
            [tech.v3.libs.buffered-image :as bufimg]
            [tech.v3.parallel.for :as pfor]
            [tvm-clj.ast :as ast]
            [tvm-clj.impl.fns.topi :as topi-fns]
            [tvm-clj.impl.base :as tvm-base]
            [tvm-clj.ast.elemwise-op :as ast-op]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tvm-clj.module :as module]
            [tvm-clj.device :as device]
            [tvm-clj.impl.fns.tvm.contrib.sort :as contrib-sort]
            [tvm-clj.impl.fns.te :as te-fns]
            [primitive-math :as pmath]
            [tech.v3.resource :as resource]
            [clojure.tools.logging :as log])
  (:import [java.util Random List Map$Entry]
           [java.util.function Consumer LongConsumer]
           [tvm_clj.impl.base TVMFunction]
           [tech.v3.datatype DoubleReader Buffer IndexReduction
            Consumers$StagedConsumer NDBuffer LongReader
            ArrayHelpers]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defonce  src-image (bufimg/load "test/data/jen.jpg"))
(defonce src-shape (dtype/shape src-image))
;;Make a 2d matrix out of the image.
(defonce src-input (dtt/clone (dtt/reshape src-image
                                           [(* (long (first src-shape))
                                               (long (second src-shape)))
                                            (last src-shape)])
                              :container-type :native-heap))


(defn- seed->random
  ^Random [seed]
  (cond
    (number? seed)
    (Random. (int seed))
    (instance? Random seed)
    seed
    (nil? seed)
    (Random.)
    :else
    (errors/throwf "Unrecognized seed type: %s" seed)))


(defmacro row-center-distance
  [dataset centers row-idx center-idx n-cols]
  `(-> (loop [col-idx# 0
              sum# 0.0]
         (if (< col-idx# ~n-cols)
           (let [diff# (pmath/- (.ndReadDouble ~dataset ~row-idx col-idx#)
                                (.ndReadDouble ~centers ~center-idx col-idx#))]
             (recur (unchecked-inc col-idx#)
                    (pmath/+ sum# (pmath/* diff# diff#))))
           sum#))
       (double)))


(defn- double-binary-search
  "Method copied from tech.v2.datatype"
  ^long [data target]
  (let [data (dtype/->buffer data)
        target (double target)
        n-elems (dtype/ecount data)]
    (loop [low (long 0)
           high n-elems]
      (if (< low high)
        (let [mid (+ low (quot (- high low) 2))
              buf-data (.readDouble data mid)
              compare-result (Double/compare buf-data target)]
          (if (= 0 compare-result)
            (recur mid mid)
            (if (and (< compare-result 0)
                     (not= mid low))
              (recur mid high)
              (recur low mid))))
        (let [buf-data (.readDouble data low)]
          (if (<= (Double/compare buf-data target) 0)
            low
            (unchecked-inc low)))))))


(defn choose-centers++
  "Implementation of the kmeans++ center choosing algorithm.  Distance-fn takes
  three arguments: dataset, centers, and distances and must mutably write
  it's result into distances."
  [dataset n-centers distance-fn {:keys [seed]}]
  (let [[n-rows n-cols] (dtype/shape dataset)
        centers (dtt/new-tensor [n-centers n-cols]
                                :container-type :native-heap
                                :datatype :float64
                                :resource-type :auto)]
    (resource/stack-resource-context
     (let [random (seed->random seed)
           n-rows (long n-rows)
           distances (dtt/new-tensor [n-rows]
                                     :container-type :native-heap
                                     :datatype :float64
                                     :resource-type :auto)
           ;;We use TVM to create an array
           scan-distances (dtt/new-tensor [n-rows]
                                          :container-type :native-heap
                                          :datatype :float64
                                          :resource-type :auto)
           initial-seed-idx (.nextInt random (int n-rows))
           _ (dtt/mset! centers 0 (dtt/mget dataset initial-seed-idx))
           n-centers (long n-centers)]
       (dotimes [idx (dec n-centers)]
         (distance-fn dataset centers idx distances scan-distances)
         (let [next-flt (.nextDouble ^Random random)
               ;;No one (not intel, not smile) actually sorts the distances
               ;;_ (contrib-sort/argsort distances indexes 0 false)
               n-rows (dtype/ecount distances)
               distance-sum (double (scan-distances (dec n-rows)))
               target-amt (* next-flt distance-sum)
               next-center-idx (double-binary-search scan-distances target-amt)]
           #_(log/infof "center chosen: %d\n %e <= %e <= %e\n %s"
                        next-center-idx
                        (scan-distances next-center-idx)
                        target-amt
                        (scan-distances (inc next-center-idx))
                        (vec (take 10 distances)))
           (dtt/mset! centers (inc idx) (dtt/mget dataset next-center-idx))))))
    centers))


(defn tvm-dist-sum-algo
  "Update the distances with values from the new centers.
  The recalculate the cumulative sum vector."
  [n-cols]
  (let [n-centers (ast/variable "n_centers")
        n-rows (ast/variable "nrows")
        n-cols (ast-op/const n-cols :int32)
        center-idx (ast/variable "center-idx")
        ;;The distance calculation is the only real issue here.
        ;;Everything else, sort, etc. is pretty quick and sorting
        centers (ast/placeholder [n-centers n-cols] "centers" :dtype :float64)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :uint8)
        ;;distances are doubles so summation is in double space
        distances (ast/placeholder [n-rows] "distances" :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-cols]
                                (ast/tvm-fn
                                 [row-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centers [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))

        expanded-distances-op (ast/compute
                               [n-rows]
                               (ast/tvm-fn
                                [row-idx]
                                (ast/commutative-reduce
                                 (ast/tvm-fn->commutative-reducer
                                  (ast/tvm-fn
                                   [sum sq-elem]
                                   (ast-op/+ sum sq-elem))
                                  [(double 0.0)])
                                 [{:domain [0 n-cols] :name "col-idx"}]
                                 [(fn [col-idx]
                                    (ast/tget squared-diff [row-idx col-idx]))]))
                               "expanded-distances")
        expanded-distances (first (ast/output-tensors expanded-distances-op))
        mindistances-op (ast/compute
                         [n-rows]
                         (ast/tvm-fn
                          [row-idx]
                          (ast/tvm-let
                           [prev-dist (ast-op/select (ast-op/eq center-idx 0)
                                                     (ast-op/max-value :float64)
                                                     (ast/tget distances [row-idx]))
                            cur-dist (-> (ast/tget expanded-distances [row-idx])
                                         (ast-op/cast :float64))]
                           (ast-op/select (ast-op/<= cur-dist prev-dist)
                                          cur-dist prev-dist)))
                         "mindistances")
        mindistances (first (ast/output-tensors mindistances-op))
        ;;nrows timestamps by n-rows results
        scan-state (ast/placeholder [n-rows] "scan_state" :dtype :float64)
        ;;Init at timestamp 0
        scan-op (ast/scan
                 ;;Initial op.  Outermost dim must be 1, describes the algorithm at
                 ;;timestate 0
                 (ast/compute [1]
                              (ast/tvm-fn [row-idx]
                                          (ast/tget mindistances [row-idx]))
                              "scan_init")
                 ;;update op, describes transition from previous timestates to current ts
                 (ast/compute [n-rows]
                              (ast/tvm-fn [ts-idx]
                                          (ast-op/+
                                           ;;grab stage from ts-1
                                           (ast/tget scan-state [(ast-op/- ts-idx (int 1))])
                                           ;;add to incoming values
                                           (ast/tget mindistances [ts-idx])))
                              "scan_update")
                 ;;State of scan algorithm.  Must have enough dimensions for each
                 ;;timestep as well as result
                 scan-state
                 ;;incoming values
                 [mindistances]
                 {:name "distance_scan"})
        scan-result (first (ast/output-tensors scan-op))
        schedule (schedule/create-schedule scan-op)
        stage-map (:stage_map schedule)
        sq-diff-stage (get stage-map squared-differences-op)
        exp-diff-stage (get stage-map expanded-distances-op)
        exp-diff-axis (last (:axis expanded-distances-op))
        mindist-stage (get stage-map mindistances-op)
        mindist-axis (last (:axis mindistances-op))
        scan-stage (get stage-map scan-op)
        scan-axis (:scan_axis scan-op)]
    (schedule/stage-compute-at sq-diff-stage exp-diff-stage exp-diff-axis)
    (schedule/stage-compute-at exp-diff-stage mindist-stage mindist-axis)
    (schedule/stage-parallel mindist-stage mindist-axis)
    {:arguments [dataset centers center-idx distances mindistances scan-result]
     :schedule schedule}))


(def tvm-dist-sum-fn*
  (delay
    (let [tvm-fn (compiler/ir->fn (tvm-dist-sum-algo 3) "dist_sum")]
      (fn [dataset centers center-idx distances scan-distances]
        (tvm-fn dataset centers center-idx distances distances scan-distances)))))


(comment
  (def distances (dtt/new-tensor [n-rows]
                                 :datatype :float64
                                 :container-type :native-heap))
  (def scan-distances (dtt/new-tensor [n-rows]
                                      :datatype :float64
                                      :container-type :native-heap))
  (def sum (dtt/new-tensor [1]
                           :datatype :float64
                           :container-type :native-heap))
  (dtype/set-constant! scan-distances 0)

  (time (@tvm-dist-sum-fn* src-input (dtt/new-tensor [1 3] :datatype :float32 :container-type :native-heap)
         0 distances scan-distances))

  (def centers (time (choose-centers++ src-input 5 @tvm-dist-sum-fn* {:seed 5})))

  )


(defrecord AggReduceContext [^doubles center
                             ^doubles score
                             ^longs n-rows])


(defn jvm-agg
  [^NDBuffer dataset ^NDBuffer center-indexes ^NDBuffer distances
   n-centers]
  (let [n-centers (long n-centers)
        [n-rows n-cols] (dtype/shape dataset)
        n-rows (long n-rows)
        n-cols (long n-cols)
        make-reduce-context #(->AggReduceContext (double-array n-cols)
                                                 (double-array 1)
                                                 (long-array 1))
        dataset-buf (dtype/->buffer dataset)
        ;;A compute tensor's inline implementation beats the tensor's
        ;;generalized implementation if you know your dimensions
        dataset (dtt/typed-compute-tensor :float64 [n-rows n-cols]
                                          [row-idx col-idx]
                                          (.readDouble dataset-buf
                                                       (pmath/+ (* row-idx n-cols)
                                                                col-idx)))
        ;;Because the number of centers is small compared to the number of rows
        ;;, the ordered reduction is faster due to much less locking and a
        ;;free merge step.
        agg-map
        (->> (dtype-reduce/ordered-group-by-reduce
              (reify IndexReduction
                (reduceIndex [this batch ctx row-idx]
                  (let [^AggReduceContext ctx (or ctx (make-reduce-context))]
                    (dotimes [col-idx n-cols]
                      (ArrayHelpers/accumPlus ^doubles (.center ctx) col-idx
                                              (.ndReadDouble dataset row-idx col-idx)))
                    (ArrayHelpers/accumPlus ^doubles (.score ctx) 0 (.ndReadDouble distances row-idx))
                    (ArrayHelpers/accumPlus ^longs (.n-rows ctx) 0 1)
                    ctx))
                (reduceReductions [this lhsCtx rhsCtx]
                  (let [^AggReduceContext lhsCtx lhsCtx
                        ^AggReduceContext rhsCtx rhsCtx]
                    (dotimes [col-idx n-cols]
                      (ArrayHelpers/accumPlus ^doubles (.center lhsCtx) col-idx
                                              (aget ^doubles (.center rhsCtx) col-idx)))
                    (ArrayHelpers/accumPlus ^doubles (.score lhsCtx) 0 (aget ^doubles (.score rhsCtx) 0))
                    (ArrayHelpers/accumPlus ^longs (.n-rows lhsCtx) 0 (aget ^longs (.n-rows rhsCtx) 0))
                    lhsCtx)))
              center-indexes)
             (map (fn [^Map$Entry entry]
                    [(.getKey entry) (.getValue entry)]))
             (sort-by first)
             (map second))
        new-centers (dtt/->tensor (mapv :center agg-map) :datatype :float64)
        row-counts (long-array (mapv (comp first :n-rows) agg-map))
        scores (double-array (mapv (comp first :score) agg-map))]
    {:new-centers (dfn// new-centers (-> (dtt/reshape row-counts [n-centers 1])
                                         (dtt/broadcast  [n-centers n-cols])))
     :row-counts row-counts
     :score (dfn/sum (dfn// scores row-counts))}))


(defn tvm-centers-distances-algo
  [n-cols]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-centers (ast/variable "n-centers")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :uint8)
        centers (ast/placeholder [n-centers n-cols] :centers :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-centers n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centers [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centers]
                               (ast/tvm-fn
                                [row-idx center-idx]
                                (ast/commutative-reduce
                                 (ast/tvm-fn->commutative-reducer
                                  (ast/tvm-fn
                                   [sum sq-elem]
                                   (ast-op/+ sum sq-elem))
                                  [(double 0.0)])
                                 [{:domain [0 n-cols] :name "col-idx"}]
                                 [(fn [col-idx]
                                    (ast/tget squared-diff [row-idx center-idx col-idx]))]))
                               "expanded-distances")
        expanded-distances (first (ast/output-tensors expanded-distances-op))
        center-indexes-assigned (topi-fns/argmin expanded-distances -1 false)
        mindistance-assign-op (:op center-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [center-indexes mindistances] (ast/output-tensors mindistance-op)

        [exp-dist-rows exp-dist-cent] (:axis expanded-distances-op)
        [mindist-rows] (get mindistance-op :axis)

        schedule (schedule/create-schedule mindistance-op)
        stage-map (:stage_map schedule)
        sq-diff-stage (stage-map squared-differences-op)
        exp-dist-stage (stage-map expanded-distances-op)
        mindist-stage (stage-map mindistance-op)]
    (schedule/stage-compute-at sq-diff-stage exp-dist-stage exp-dist-cent)
    (schedule/stage-compute-at exp-dist-stage mindist-stage mindist-rows)
    (schedule/stage-parallel mindist-stage mindist-rows)
    {:schedule schedule
     :arguments [dataset centers center-indexes mindistances]}))


(defn make-tvm-centers-distances-fn
  [n-cols]
  (-> (tvm-centers-distances-algo n-cols)
      (compiler/ir->fn "cpu_centers_distances")))


(def tvm-centers-distances-fn* (delay (make-tvm-centers-distances-fn 3)))

(defn jvm-tvm-iterate-kmeans
  [dataset centers]
  (resource/stack-resource-context
   (let [[n-rows n-cols] (dtype/shape dataset)
         [n-centers n-cols] (dtype/shape centers)
         center-indexes (dtt/new-tensor [n-rows]
                                        :datatype :int32
                                        :container-type :native-heap
                                        :resource-type :auto)
         distances (dtt/new-tensor [n-rows]
                                   :datatype :float64
                                   :container-type :native-heap
                                   :resource-type :auto)]
     (@tvm-centers-distances-fn* dataset centers
      center-indexes distances)
     (jvm-agg dataset center-indexes distances n-centers))))



(defn tvm-brute-force-algo
  "brute force aggregate centers.  Each thread gets one centroid element to agg into."
  [n-cols]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-centers (ast/variable "n-centers")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :uint8)
        centers (ast/placeholder [n-centers n-cols] :centers :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-centers n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centers [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centers]
                               (ast/tvm-fn
                                [row-idx center-idx]
                                (ast/commutative-reduce
                                 (ast/tvm-fn->commutative-reducer
                                  (ast/tvm-fn
                                   [sum sq-elem]
                                   (ast-op/+ sum sq-elem))
                                  [(double 0.0)])
                                 [{:domain [0 n-cols] :name "col-idx"}]
                                 [(fn [col-idx]
                                    (ast/tget squared-diff [row-idx center-idx col-idx]))]))
                               "expanded-distances")
        expanded-distances (first (ast/output-tensors expanded-distances-op))
        center-indexes-assigned (topi-fns/argmin expanded-distances -1 false)
        mindistance-assign-op (:op center-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [center-indexes mindistances] (ast/output-tensors mindistance-op)
        agg-op (ast/compute
                [n-centers n-cols]
                (ast/tvm-fn
                 [center-idx col-idx]
                 (ast/commutative-reduce
                  (ast/tvm-fn->commutative-reducer
                   (ast/tvm-fn
                    [fsum count-acc score-acc sq-elem count-elem score-elem]
                    [(ast-op/+ fsum sq-elem)
                     (ast-op/+ count-acc count-elem)
                     (ast-op/+ score-elem score-acc)])
                   [(double 0.0)
                    (int 0)
                    (double 0.0)])
                  [{:domain [0 n-rows] :name "row-idx"}]
                  [(fn [row-idx]
                     [(ast-op/cast (ast/tget dataset [row-idx col-idx])
                                   :float64)
                      (ast-op/const (int 1))
                      (ast-op/cast (ast/tget mindistances [row-idx])
                                   :float64)])]
                  (fn [row-idx]
                    (ast-op/eq center-idx (ast/tget center-indexes [row-idx])))))
                "new-centers-sum")
        [new-centers new-counts new-scores] (ast/output-tensors agg-op)


        schedule (schedule/create-schedule agg-op)
        stage-map (:stage_map schedule)
        sq-diff-stage (stage-map squared-differences-op)
        exp-dist-stage (stage-map expanded-distances-op)
        mindist-copy-stage (stage-map mindistance-op)
        ;; mindist-calc-op (:op (first (ast/input-tensors mindistance-op)))
        ;; mindist-calc-stage (stage-map mindist-calc-op)
        ;;[mindist-calc-rows] (get-in mindist-calc-stage [:op :axis])
        [mindist-copy-rows] (get-in mindist-copy-stage [:op :axis])
        [exp-dist-rows exp-dist-cent] (:axis expanded-distances-op)
        agg-stage (get stage-map agg-op)
        [center-axis col-axis] (:axis agg-op)]
    (schedule/stage-compute-at sq-diff-stage exp-dist-stage exp-dist-cent)
    (schedule/stage-compute-at exp-dist-stage mindist-copy-stage mindist-copy-rows)
    ;; (schedule/stage-compute-at mindist-calc-stage mindist-copy-stage mindist-copy-rows)
    ;; (schedule/stage-vectorize mindist-copy-stage mindist-copy-rows)

    ;;Two parallelized passes over the data per iteration
    (schedule/stage-parallel mindist-copy-stage mindist-copy-rows)

    (schedule/stage-parallel agg-stage center-axis)

    {:arguments [dataset centers new-scores new-counts new-centers]
     :schedule schedule}))


(def tvm-all-in-one* (delay
                       (-> (tvm-brute-force-algo 3)
                           (compiler/ir->fn "tvm_all_in_one"))))


(defn tvm-all-in-one-iterate-kmeans
  [dataset centers]
  (resource/stack-resource-context
   (let [[n-rows n-cols] (dtype/shape dataset)
         [n-centers n-cols] (dtype/shape centers)
         center-indexes (dtt/new-tensor [n-rows]
                                        :datatype :int32
                                        :container-type :native-heap
                                        :resource-type :auto)
         distances (dtt/new-tensor [n-rows]
                                   :datatype :float32
                                   :container-type :native-heap
                                   :resource-type :auto)
         new-centers (dtt/new-tensor [n-centers n-cols]
                                     :datatype :float64
                                     :container-type :native-heap
                                     :resource-type :auto)
         new-scores (dtt/new-tensor [n-centers n-cols]
                                    :datatype :float64
                                    :container-type :native-heap
                                    :resource-type :auto)
         new-counts (dtt/new-tensor [n-centers n-cols]
                                    :datatype :int32
                                    :container-type :native-heap
                                    :resource-type :auto)]
     (@tvm-all-in-one* dataset centers new-scores new-counts new-centers)
     (let [row-counts (long-array (dtt/select new-counts :all 0))]
       {:new-centers (dtype/clone (dfn// new-centers
                                         (-> (dtt/reshape row-counts [n-centers 1])
                                             (dtt/broadcast [n-centers n-cols]))))
        :row-counts row-counts
        :score (dfn/sum (dfn// (dtt/select new-scores :all 0) row-counts))}))))


(defn tvm-assign-centers-algo
  [n-cols]
  (let [n-cols (ast-op/const n-cols :int32)
        n-rows (ast/variable "n-rows")
        n-centers (ast/variable "n-centers")
        centers (ast/placeholder [n-centers n-cols] "centers" :dtype :float64)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :uint8)
        squared-differences-op (ast/compute
                                [n-rows n-centers n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centers [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centers]
                               (ast/tvm-fn
                                [row-idx center-idx]
                                (ast/commutative-reduce
                                 (ast/tvm-fn->commutative-reducer
                                  (ast/tvm-fn
                                   [sum sq-elem]
                                   (ast-op/+ sum sq-elem))
                                  [(double 0.0)])
                                 [{:domain [0 n-cols] :name "col-idx"}]
                                 [(fn [col-idx]
                                    (ast/tget squared-diff [row-idx center-idx col-idx]))]))
                               "expanded-distances")
        expanded-distances (first (ast/output-tensors expanded-distances-op))
        center-indexes-assigned (topi-fns/argmin expanded-distances -1 false)
        mindistance-assign-op (:op center-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [center-indexes mindistances] (ast/output-tensors mindistance-op)
        result-op (ast/compute
                   [n-rows n-cols]
                   (ast/tvm-fn
                    [row-idx col-idx]
                    (-> (ast/tget centers
                                  [(ast/tget center-indexes [row-idx])
                                   col-idx])
                        (ast-op/cast :uint8)))
                   "result-ds")
        schedule (schedule/create-schedule result-op)
        stage-map (:stage_map schedule)
        result (first (ast/output-tensors result-op))]
    (schedule/stage-compute-at (stage-map squared-differences-op)
                               (stage-map expanded-distances-op)
                               (last (:axis expanded-distances-op)))
    (schedule/stage-compute-at (stage-map expanded-distances-op)
                               (stage-map mindistance-op)
                               (last (:axis mindistance-op)))
    (schedule/stage-compute-at (stage-map mindistance-op)
                               (stage-map result-op)
                               (last (:axis result-op)))
    (schedule/stage-parallel (stage-map result-op)
                             (first (:axis result-op)))
    {:schedule schedule
     :arguments [dataset centers result]}))


(def assign-clusters-fn* (delay
                          (-> (tvm-assign-centers-algo 3)
                              (compiler/ir->fn "assign_clusters"))))


(defn quantize-image
  [src-path n-quantization n-iters seed]
  (resource/stack-resource-context
   (let [src-img (bufimg/load src-path)
         [height width channels] (dtype/shape src-img)
         n-rows (* (long height) (long width))
         dataset (-> (dtt/reshape src-img [n-rows channels])
                     (dtt/clone :container-type :native-heap
                                :resource-type :stack))
         centers (time (choose-centers++ dataset n-quantization @tvm-dist-sum-fn*
                                         {:seed 6}))
         scores (time (mapv (fn [idx]
                              (log/infof "Iteration %d" idx)
                              (let [{:keys [new-centers score]}
                                    (jvm-tvm-iterate-kmeans dataset centers)]
                                (dtype/copy! new-centers centers)
                                score))
                            (range n-iters)))
         result-img (bufimg/new-image height width (bufimg/image-type src-img))
         result-tens (dtt/new-tensor (dtype/shape dataset)
                                     :datatype (dtype/elemwise-datatype src-img)
                                     :container-type :native-heap
                                     :resource-type :stack)]
     (@assign-clusters-fn* dataset centers result-tens)
     (log/infof "Scores: %s" scores)
     {:centers (dtype/clone centers)
      :result (dtype/copy! result-tens result-img)
      :scores scores})))


(comment
  (def jvm-tvm (time (jvm-tvm-iterate-kmeans src-input centers)))
  (def tvm-allinone (time (tvm-all-in-one-iterate-kmeans src-input centers)))
  (dotimes [iter 10]
    (let [n-quantization (* (+ iter 1) 5)]
      (log/infof "Quantization: %d" n-quantization)
      (-> (quantize-image "test/data/castle.jpg" n-quantization 3 6)
          (:result)
          (bufimg/save! (format "quantized-%d.png" n-quantization)))))
  )
