(ns tvm-clj.application.kmeans
  "High performance implementation of the KMeans algorithm"
  (:require [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.reductions :as dtype-reduce]
            [tech.v3.datatype.nio-buffer :as nio-buffer]
            [tech.v3.datatype.errors :as errors]
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

(comment
  (do
    (defonce  src-image (bufimg/load "test/data/jen.jpg"))
    (defonce src-shape (dtype/shape src-image))
    ;;Make a 2d matrix out of the image.
    (defonce src-input (dtt/clone (dtt/reshape src-image
                                               [(* (long (first src-shape))
                                                   (long (second src-shape)))
                                                (last src-shape)])
                                  :container-type :native-heap)))
  )


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


(defn- choose-centroids++
  "Implementation of the kmeans++ center choosing algorithm.  Distance-fn takes
  three arguments: dataset, centroids, and distances and must mutably write
  it's result into distances."
  [dataset n-centroids distance-fn {:keys [seed]}]
  (let [[n-rows n-cols] (dtype/shape dataset)
        centroids (dtt/new-tensor [n-centroids n-cols]
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
           _ (dtt/mset! centroids 0 (dtt/mget dataset initial-seed-idx))
           n-centroids (long n-centroids)
           last-idx (dec n-rows)]
       (dotimes [idx (dec n-centroids)]
         (distance-fn dataset centroids idx distances distances scan-distances)
         (let [next-flt (.nextDouble ^Random random)
               ;;No one (not intel, not smile) actually sorts the distances
               ;;_ (contrib-sort/argsort distances indexes 0 false)
               n-rows (dtype/ecount distances)
               distance-sum (double (scan-distances (dec n-rows)))
               target-amt (* next-flt distance-sum)
               next-center-idx (min last-idx
                                    ;;You want the one just *after* where you could safely insert
                                    ;;the distance as the next distance is likely much larger than the
                                    ;;current distance and thus your probability of getting a vector that
                                    ;;that is a large distance away than any known vectors is higher
                                    (inc (double-binary-search scan-distances target-amt)))]
           #_(log/infof "center chosen: %d\n %e <= %e <= %e\n %s"
                        next-center-idx
                        (scan-distances next-center-idx)
                        target-amt
                        (scan-distances (inc next-center-idx))
                        (vec (take 10 distances)))
           (dtt/mset! centroids (inc idx) (dtt/mget dataset next-center-idx))))))
    centroids))


(defn- tvm-dist-sum-algo
  "Update the distances with values from the new centroid and produce a cumulative
  summation vector we can binary search through.  This uses a special form where we just
  add in the new centroid to our existing distance vector.  This is similar to the
  distance methods below except we only calculate one centroid at a time."
  [n-cols dataset-datatype]
  (let [n-centroids (ast/variable "n_centroids")
        n-rows (ast/variable "nrows")
        n-cols (ast-op/const n-cols :int32)
        center-idx (ast/variable "center-idx")
        ;;The distance calculation is the only real issue here.
        ;;Everything else, sort, etc. is pretty quick and sorting
        centroids (ast/placeholder [n-centroids n-cols] "centroids" :dtype :float64)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype dataset-datatype)
        ;;distances are doubles so summation is in double space
        distances (ast/placeholder [n-rows] "distances" :dtype :float64)
        squared-differences-op
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
    {:arguments [dataset centroids center-idx distances mindistances scan-result]
     :schedule schedule}))


(def ^:private make-tvm-dist-sum-fn
  (memoize
   (fn [n-cols dataset-datatype]
     (compiler/ir->fn (tvm-dist-sum-algo n-cols dataset-datatype) "dist_sum"))))


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

  (time ((make-tvm-dist-sum-fn 3 :uint8) src-input (dtt/new-tensor [1 3] :datatype :float32 :container-type :native-heap)
         0 distances scan-distances))

  (def centroids (time (choose-centroids++ src-input 5 (make-tvm-dist-sum-fn 3 :uint8)
                                       {:seed 5})))

  )


(defrecord AggReduceContext [^doubles center
                             ^doubles score
                             ^longs n-rows])


(defn- jvm-agg
  "Aggregate assigned rows into new centroids.  Works via
  `tech.v3.datatype.argops/arggroup`."
  [^NDBuffer dataset ^NDBuffer centroid-indexes ^NDBuffer distances
   n-centroids]
  (let [n-centroids (long n-centroids)
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
        ;;Because the number of centroids is small compared to the number of rows
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
              centroid-indexes)
             (map (fn [^Map$Entry entry]
                    [(.getKey entry) (.getValue entry)]))
             (sort-by first)
             (map second))
        new-centroids (dtt/->tensor (mapv :center agg-map) :datatype :float64)
        row-counts (long-array (mapv (comp first :n-rows) agg-map))
        scores (double-array (mapv (comp first :score) agg-map))]
    {:new-centroids (dfn// new-centroids (-> (dtt/reshape row-counts [n-centroids 1])
                                         (dtt/broadcast  [n-centroids n-cols])))
     :row-counts row-counts
     ;;Score *before* this iteration calculated during course of this iteration.
     :score (dfn// (dfn/sum scores)
                   (dfn/sum row-counts))}))


(defn- ast-centroid-distances
  "Compute the full square distances matrix."
  [dataset centroids]
  (let [[n-rows n-cols] (dtype/shape dataset)
        [n-centroids n-cols] (dtype/shape centroids)
        squared-differences-op (ast/compute
                                [n-rows n-centroids n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centroids
                                                         [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))]
    (-> (ast/compute
         [n-rows n-centroids]
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
        (ast/output-tensors)
        (first))))


(defn- schedule-squared-distances
  [schedule sq-diff-tensor]
  (let [stage-map (:stage_map schedule)
        sq-diff-op (:op sq-diff-tensor)
        exp-diff-tens (first (ast/input-tensors sq-diff-op))
        exp-diff-op (:op exp-diff-tens)]
    (schedule/inline-op schedule exp-diff-op sq-diff-op -1)))


(defn- tvm-centroids-distances-algo
  [n-cols dataset-datatype]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-centroids (ast/variable "n-centroids")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype dataset-datatype)
        centroids (ast/placeholder [n-centroids n-cols] :centroids :dtype :float64)
        sq-diff-tensor (ast-centroid-distances dataset centroids)
        centroid-indexes-assigned (topi-fns/argmin sq-diff-tensor -1 false)
        mindistance-assign-op (:op centroid-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [centroid-indexes mindistances] (ast/output-tensors mindistance-op)
        sq-diff-op (:op sq-diff-tensor)
        schedule (schedule/create-schedule mindistance-op)
        stage-map (:stage_map schedule)]
    (schedule-squared-distances schedule sq-diff-tensor)
    (schedule/inline-op sq-diff-op mindistance-op 0)
    ()
    (schedule/stage-parallel (stage-map mindistance-op)
                             (first (:axis mindist-op)))
    {:schedule schedule
     :arguments [dataset centroids centroid-indexes mindistances]}))


(def ^:private make-tvm-centroids-distances-fn
  (memoize
   (fn [n-cols dataset-datatype]
     (-> (tvm-centroids-distances-algo n-cols dataset-datatype)
         (compiler/ir->fn "cpu_centroids_distances")))))


(defn- jvm-tvm-iterate-kmeans!
  [dataset centroids centroid-indexes distances tvm-centroids-distance-fn]
  (let [[n-rows n-cols] (dtype/shape dataset)
        [n-centroids n-cols] (dtype/shape centroids)]
    (tvm-centroids-distance-fn dataset centroids centroid-indexes distances)
    (jvm-agg dataset centroid-indexes distances n-centroids)))



(defn- tvm-brute-force-algo
  "brute force aggregate centroids.  Each thread gets one centroid element to agg into.
  Note that I do not use this in kmeans++ or anything else but it is surprisingly fast
  for smaller numbers of cols and clusters."
  [n-cols dataset-datatype]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-centroids (ast/variable "n-centroids")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype dataset-datatype)
        centroids (ast/placeholder [n-centroids n-cols] :centroids :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-centroids n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centroids [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centroids]
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
        centroid-indexes-assigned (topi-fns/argmin expanded-distances -1 false)
        mindistance-assign-op (:op centroid-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [centroid-indexes mindistances] (ast/output-tensors mindistance-op)
        agg-op (ast/compute
                [n-centroids n-cols]
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
                    (ast-op/eq center-idx (ast/tget centroid-indexes [row-idx])))))
                "new-centroids-sum")
        [new-centroids new-counts new-scores] (ast/output-tensors agg-op)


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

    {:arguments [dataset centroids new-scores new-counts new-centroids]
     :schedule schedule}))


(def ^:private tvm-all-in-one* (delay
                                 (-> (tvm-brute-force-algo 3 :uint8)
                                     (compiler/ir->fn "tvm_all_in_one"))))


(defn- tvm-all-in-one-iterate-kmeans
  [dataset centroids]
  @tvm-all-in-one*
  (resource/stack-resource-context
   (let [[n-rows n-cols] (dtype/shape dataset)
         [n-centroids n-cols] (dtype/shape centroids)
         centroid-indexes (dtt/new-tensor [n-rows]
                                        :datatype :int32
                                        :container-type :native-heap
                                        :resource-type :auto)
         distances (dtt/new-tensor [n-rows]
                                   :datatype :float32
                                   :container-type :native-heap
                                   :resource-type :auto)
         new-centroids (dtt/new-tensor [n-centroids n-cols]
                                     :datatype :float64
                                     :container-type :native-heap
                                     :resource-type :auto)
         new-scores (dtt/new-tensor [n-centroids n-cols]
                                    :datatype :float64
                                    :container-type :native-heap
                                    :resource-type :auto)
         new-counts (dtt/new-tensor [n-centroids n-cols]
                                    :datatype :int32
                                    :container-type :native-heap
                                    :resource-type :auto)]
     (@tvm-all-in-one* dataset centroids new-scores new-counts new-centroids)
     (let [row-counts (long-array (dtt/select new-counts :all 0))]
       {:new-centroids (dtype/clone (dfn// new-centroids
                                         (-> (dtt/reshape row-counts [n-centroids 1])
                                             (dtt/broadcast [n-centroids n-cols]))))

        :row-counts row-counts
        :score (dfn/sum (dfn// (dtt/select new-scores :all 0) row-counts))}))))


(defn- ensure-native
  [ds]
  (if (dtype/as-native-buffer ds)
    ds
    (dtt/clone ds
               :container-type :native-heap
               :resource-type :auto)))


(defn- tvm-score-algo
  [n-cols ds-dtype]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-centroids (ast/variable "n-centroids")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype ds-dtype)
        centroids (ast/placeholder [n-centroids n-cols] :centroids :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-centroids n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centroids [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centroids]
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
        centroid-indexes-assigned (topi-fns/argmin expanded-distances -1 false)
        mindistance-assign-op (:op centroid-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [centroid-indexes mindistances] (ast/output-tensors mindistance-op)
        sum-op (ast/compute
                [1]
                (ast/tvm-fn
                 [idx]
                 (ast/commutative-reduce
                  (ast/tvm-fn->commutative-reducer
                   (ast/tvm-fn
                    [sum sq-elem]
                    (ast-op/+ sum sq-elem))
                   [(double 0.0)])
                  [{:domain [0 n-rows] :name "row-idx"}]
                  [(fn [row-idx]
                     (ast/tget mindistances [row-idx]))]))
                "sum")
        sum-tensor (first (ast/output-tensors sum-op))
        schedule (schedule/create-schedule sum-op)
        stage-map (:stage_map schedule)]
    (schedule/stage-compute-at (stage-map squared-differences-op)
                               (stage-map expanded-distances-op)
                               (last (:axis expanded-distances-op)))
    (schedule/stage-compute-at (stage-map expanded-distances-op)
                               (stage-map mindistance-op)
                               (last (:axis mindistance-op)))
    (schedule/stage-compute-at (stage-map mindistance-op)
                               (stage-map sum-op)
                               (last (:axis sum-op)))
    (schedule/stage-parallel (stage-map mindistance-op)
                             (first (:axis mindistance-op)))
    {:schedule schedule
     :arguments [dataset centroids sum-tensor]}))

(def ^:private make-tvm-score-fn
  (memoize
   (fn [n-cols ds-dtype]
     (-> (tvm-score-algo n-cols ds-dtype)
         (compiler/ir->fn "tvm_score")))))


(defn- precompile-kmeans-functions
  [n-cols ds-dtype]
  (let [tvm-dist-sum-fn (make-tvm-dist-sum-fn n-cols ds-dtype)
        tvm-centroids-dist-fn (make-tvm-centroids-distances-fn n-cols ds-dtype)
        score-fn (make-tvm-score-fn n-cols ds-dtype)]
    [tvm-dist-sum-fn tvm-centroids-dist-fn score-fn]))


(defn kmeans++
  "Find K cluster centroids via kmeans++ center initialization
  followed by Lloyds algorithm.
  Dataset must be a matrix (2d tensor).

  * `dataset` - 2d matrix of numeric datatype.
  * `n-centroids` - How many centroids to find.

  Returns map of:

  * `:centroids` - 2d tensor of double centroids
  * `:centroid-indexes` - 1d integer vector of assigned center indexes.
  * `:iteration-scores` - n-iters+1 length array of mean squared error scores container
    the scores from centroid assigned up to the score when the algorithm
    terminates.

  Options:

  * `:minimal-improvement-threshold` - defaults to 0.01 - algorithm terminates if
     (1.0 - error(n-1)/error(n-2)) < error-diff-threshold.  When Zero means algorithm will
     always train to max-iters.
  * `:n-iters` - defaults to 100 - Max number of iterations, algorithm terminates
     if `(>= iter-idx n-iters).
  * `:rand-seed` - integer or implementation of `java.util.Random`.
  "
  [dataset n-centroids & [{:keys [n-iters rand-seed
                                  minimal-improvement-threshold]
                           :or {minimal-improvement-threshold 0.01}
                           :as options}]]
  (errors/when-not-error
   (== 2 (dtype/ecount (dtype/shape dataset)))
   "Dataset must be a matrix of rank 2")
  (let [[n-rows n-cols] (dtype/shape dataset)
        ds-dtype (dtype/elemwise-datatype dataset)
        [tvm-dist-sum-fn tvm-centroids-dist-fn score-fn]
        (precompile-kmeans-functions n-cols ds-dtype)
        n-iters (long (or n-iters 100))
        minimal-improvement-threshold (double (or minimal-improvement-threshold 0.011))]
    (log/infof "Choosing n-centroids %d with %f improvement threshold and max %d iters"
               n-centroids minimal-improvement-threshold n-iters)
    (resource/stack-resource-context
     (let [dataset (ensure-native dataset)
           centroids (if (number? n-centroids)
                     (choose-centroids++ dataset n-centroids
                                       tvm-dist-sum-fn
                                       {:seed rand-seed})
                     (do
                       (errors/when-not-error
                        (== 2 (count (dtype/shape n-centroids)))
                        "Centroids must be rank 2")
                       (ensure-native n-centroids)))
           centroid-indexes (dtt/new-tensor [n-rows]
                                          :datatype :int32
                                          :container-type :native-heap
                                          :resource-type :auto)
           distances (dtt/new-tensor [n-rows]
                                     :datatype :float64
                                     :container-type :native-heap
                                     :resource-type :auto)

           minimal-improvement-threshold (double minimal-improvement-threshold)
           dec-n-iters (dec n-iters)
           scores (if-not (== 0 n-iters)
                    (loop [iter-idx 0
                           last-score 0.0
                           scores []]
                      (let [{:keys [new-centroids score]}
                            ;;Side effects include updating the centroids,
                            ;;center-indexes, and distances, while potentially
                            ;;calling your ex and telling them you still love them.
                            (jvm-tvm-iterate-kmeans! dataset centroids
                                                     centroid-indexes distances
                                                     tvm-centroids-dist-fn)
                            score (double score)
                            rel-score (if-not (== 0.0 last-score)
                                        (- 1.0 (/ score last-score))
                                        1.0)]
                        (dtype/copy! new-centroids centroids)
                        (log/infof "Iteration %d out of %d - relative improvement %f->%f=%f"
                                   iter-idx n-iters last-score score rel-score)
                        (if (and (< iter-idx dec-n-iters)
                                 (not= 0.0 score)
                                 (> rel-score minimal-improvement-threshold))
                          (recur (unchecked-inc iter-idx) score (conj scores score))
                          scores)))
                    [])
           score-tens (dtt/new-tensor [1]
                                      :datatype :float64
                                      :container-type :native-heap
                                      :resource-type :auto)]
       (score-fn dataset centroids score-tens)
       ;;Clone data back into jvm land to escape the resource context
       {:centroids (dtt/clone centroids)
        :centroid-indexes (dtt/clone centroid-indexes)
        :iteration-scores (vec (concat scores [(/ (double (score-tens 0))
                                                  (double n-rows))]))}))))


(defn- centroid-indexes->centroid-counts
  "Given tensor of assigned center indexes, produce an in-order
  tensor of centroid counts per center.  Array is in order of center
  index."
  ^NDBuffer [centroid-indexes & [center-offset]]
  (let [center-offset (long (or center-offset 0))])
  (->> (argops/arggroup centroid-indexes)
       (into {})
       (sort-by first)
       (map (comp (partial + center-offset) dtype/ecount second))
       (long-array)
       (dtt/ensure-tensor)))


(defn- concatenate-results
  "Given a sequence of maps, return one result map with
  tensors with one extra dimension.  Works when every result has the
  same length."
  [result-seq]
  (when (seq result-seq)
    (->> (first result-seq)
         (map (fn [[k v]]
                [k (dtt/->tensor (mapv k result-seq)
                                 :datatype (dtype/elemwise-datatype v))]))
         (into {}))))


(defn- reorder-tensor-algo
  [ds-dtype]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast/variable "n-cols")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype ds-dtype)
        indexes (ast/placeholder [n-rows] "indexes" :dtype :int32)
        reorder-op (ast/compute
                    [n-rows n-cols]
                    (ast/tvm-fn
                     [row-idx col-idx]
                     (ast/tget dataset [(ast/tget indexes [row-idx])
                                        col-idx]))
                    "result")
        result (first (ast/output-tensors reorder-op))
        schedule (schedule/create-schedule reorder-op)
        stage-map (:stage_map schedule)]
    (schedule/stage-parallel (stage-map reorder-op) (first (:axis reorder-op)))
    {:schedule schedule
     :arguments [dataset indexes result]}))


(def ^:private reorder-tensor-fn
  (memoize
   (fn [ds-dtype]
     (-> (reorder-tensor-algo ds-dtype)
         (compiler/ir->fn "reorder-tensor")))))


(defn order-data-labels
  "Order the dataset and labels such that labels are monotonically increasing.
  returns tuple of [dataset labels]"
  [data labels]
  (let [ds-dtype (dtype/elemwise-datatype data)
        label-indexes (dtt/clone (argops/argsort labels)
                                 :container-type :native-heap
                                 :resource-type :auto)
        reorder-fn (reorder-tensor-fn ds-dtype)
        data (ensure-native data)
        result (dtt/new-tensor (dtype/shape data)
                               :datatype ds-dtype
                               :resource-type :auto
                               :container-type :native-heap)]
    (reorder-fn data label-indexes result)
    [result (dtype/indexed-buffer label-indexes labels)]))


(defn train-per-label
  "Given a dataset along with per-row integer labels, train N per-label kmeans centroids
  returning a map which use can use with predict-per-label."
  [data labels n-per-label & [{:keys [input-ordered?]
                               :as options}]]
  (when-not (empty? labels)
    (resource/stack-resource-context
     ;;Organize data per-label
     (let [n-per-label (long n-per-label)
           ds-dtype (dtype/elemwise-datatype data)
           [data labels] (if input-ordered?
                           [(ensure-native data) labels]
                           ;;Order data and labels by increasing index
                           (order-data-labels data labels))
           [n-rows n-cols] (dtype/shape data)
           labels (->> (argops/arggroup labels)
                       (into {})
                       (sort-by first)
                       ;;arggroup be default uses an 'ordered' algorithm that guarantees
                       ;;the result index list is ordered.
                       (mapv (fn [[label idx-list]]
                               [label
                                [(first idx-list) (last idx-list)]])))
           n-labels (count labels)]
       (->> labels
            (map (fn [[label [^long idx-start ^long past-idx-end]]]
                   ;;Tensor selection from contiguous data of a range with an increment of 1
                   ;;is guaranteed to produce contiguous data
                   (log/infof "Training centroids for label %s" label)
                   (let [{:keys [centroids centroid-indexes iteration-scores]}
                         (-> (dtt/select data (range idx-start past-idx-end))
                             (kmeans++ n-per-label options))]
                     {:centroids centroids
                      :labels label
                      :centroid-counts (centroid-indexes->centroid-counts
                                        centroid-indexes idx-start)
                      :iteration-scores (last iteration-scores)})))
            (concatenate-results)
            (merge {:kmeans-type :n-per-label}))))))

(defn- per-label-prob-dist-algo
  [n-cols ds-dtype]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-labels (ast/variable "n-labels")
        n-centroids (ast/variable "n-centroids")
        n-per-label (ast-op// n-centroids n-labels)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype ds-dtype)
        centroids (ast/placeholder [n-centroids n-cols] "centroids" :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-centroids n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centroids [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centroids]
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
        per-label-sum-op (ast/compute
                          [n-rows n-labels]
                          (ast/tvm-fn
                           [row-idx label-idx]
                           (ast/commutative-reduce
                            (ast/tvm-fn->commutative-reducer
                             (ast/tvm-fn
                              [sum sq-elem]
                              (ast-op/min sum sq-elem))
                             [(ast-op/max-value :float64)])
                            [{:domain [0 n-per-label] :name "per-label-idx"}]
                            [(fn [per-label-idx]
                               (ast/tget expanded-distances [row-idx
                                                             (ast-op/+
                                                              (ast-op/* label-idx n-per-label)
                                                              per-label-idx)]))]))
                          "per-label-sum")
        per-label-sum (first (ast/output-tensors per-label-sum-op))
        per-row-sum-op (ast/compute
                        [n-rows]
                        (ast/tvm-fn
                         [row-idx]
                         (ast/commutative-reduce
                          (ast/tvm-fn->commutative-reducer
                           (ast/tvm-fn
                            [sum sq-elem]
                            (ast-op/+ sum sq-elem))
                           [(double 0.0)])
                          [{:domain [0 n-labels] :name "label-idx"}]
                          [(fn [label-idx]
                             (ast/tget per-label-sum [row-idx label-idx]))]))
                        "per-row-summation")
        ;;We need to invert the probability distribution so the shortest distance
        ;;gets the highest probability
        prob-divisor (ast-op/max 1.0 (ast-op/- (ast-op/cast n-labels :float64) 1.0))
        per-row-sums (first (ast/output-tensors per-row-sum-op))
        prob-dist-op (ast/compute
                      [n-rows n-labels]
                      (ast/tvm-fn
                       [row-idx label-idx]
                       (ast-op//
                        (ast-op/- 1.0
                                  (ast-op// (ast/tget per-label-sum [row-idx label-idx])
                                            (ast/tget per-row-sums [row-idx])))
                        prob-divisor))
                      "prob-dist")
        prob-dist (first (ast/output-tensors prob-dist-op))
        assigned-indexes-op (-> (topi-fns/argmax prob-dist -1 false)
                                (:op)
                                (ast/input-tensors)
                                (first)
                                (:op))
        assigned-indexes (first (ast/output-tensors assigned-indexes-op))

        schedule (schedule/create-schedule assigned-indexes-op)
        stage-map (:stage_map schedule)]
    (schedule/stage-compute-at (stage-map squared-differences-op)
                               (stage-map expanded-distances-op)
                               (last (:axis expanded-distances-op)))
    (schedule/stage-compute-at (stage-map expanded-distances-op)
                               (stage-map per-label-sum-op)
                               (last (:axis per-label-sum-op)))

    (schedule/stage-parallel (stage-map per-label-sum-op)
                             (first (:axis per-label-sum-op)))
    (schedule/stage-compute-at (stage-map per-row-sum-op)
                               (stage-map prob-dist-op)
                               (first (:axis prob-dist-op)))
    (schedule/stage-compute-at (stage-map prob-dist-op)
                               (stage-map assigned-indexes-op)
                               (first (:axis assigned-indexes-op)))
    (schedule/stage-parallel (stage-map assigned-indexes-op)
                             (first (:axis assigned-indexes-op)))

    {:schedule schedule
     :arguments [dataset centroids n-labels prob-dist assigned-indexes]}))


(def ^:private make-prob-dist-fn
  (memoize
   (fn [n-cols ds-dtype]
     (-> (per-label-prob-dist-algo n-cols ds-dtype)
         (compiler/ir->fn "per-label-prob-dist")))))


(defn predict-per-label
  "Return both a probability distribution per row across each label and
  a 1d tensor of assigned label indexes.
  "
  [data model]
  (let [prob-dist-fn (make-prob-dist-fn (last (dtype/shape data))
                                        (dtype/elemwise-datatype data))]
    (resource/stack-resource-context
     (let [{:keys [centroids labels]} model
           [n-labels n-per-label n-cols] (dtype/shape centroids)
           [n-rows n-data-cols] (dtype/shape data)
           _ (errors/when-not-errorf
              (= n-cols n-data-cols)
              "Data (%d), model (%d) have different feature counts"
              n-data-cols n-cols)
           data (ensure-native data)
           n-centroids (* (long n-labels)
                          (long n-per-label))
           centroids (-> (dtt/reshape centroids [n-centroids n-cols])
                         (ensure-native))
           indexes (dtt/new-tensor [n-rows]
                                   :datatype :int32
                                  :container-type :native-heap
                                  :resource-type :auto)
           result (dtt/new-tensor [n-rows n-labels]
                                  :datatype :float64
                                  :container-type :native-heap
                                  :resource-type :auto)]
       (prob-dist-fn data centroids n-labels result indexes)
       {:probability-distribution (dtype/clone result)
        :label-indexes (dtype/clone indexes)}))))


(defn- tvm-assign-centroids-algo
  [n-cols dataset-datatype]
  (let [n-cols (ast-op/const n-cols :int32)
        n-rows (ast/variable "n-rows")
        n-centroids (ast/variable "n-centroids")
        centroids (ast/placeholder [n-centroids n-cols] "centroids" :dtype :float64)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype dataset-datatype)
        squared-differences-op (ast/compute
                                [n-rows n-centroids n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (-> (ast/tget dataset [row-idx col-idx])
                                                (ast-op/cast :float64))
                                   center-elem (ast/tget centroids [center-idx col-idx])
                                   diff (ast-op/- row-elem center-elem)]
                                  (ast-op/* diff diff)))
                                "squared-diff")
        squared-diff (first (ast/output-tensors squared-differences-op))
        expanded-distances-op (ast/compute
                               [n-rows n-centroids]
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
        centroid-indexes-assigned (topi-fns/argmin expanded-distances -1 false)
        mindistance-assign-op (:op centroid-indexes-assigned)
        mindistance-op (:op (first (ast/input-tensors mindistance-assign-op)))
        [centroid-indexes mindistances] (ast/output-tensors mindistance-op)
        result-op (ast/compute
                   [n-rows n-cols]
                   (ast/tvm-fn
                    [row-idx col-idx]
                    (-> (ast/tget centroids
                                  [(ast/tget centroid-indexes [row-idx])
                                   col-idx])
                        (ast-op/cast dataset-datatype)))
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
     :arguments [dataset centroids result]}))


(def ^:private make-assign-clusters-fn
  (memoize
   (fn [n-cols ds-type]
     (-> (tvm-assign-centroids-algo n-cols ds-type)
         (compiler/ir->fn "assign_clusters")))))


(defn quantize-image
  [src-path dst-path n-quantization & [{:keys [n-iters seed]
                                        :or {n-iters 5}}]]
  (let [src-img (bufimg/load src-path)
        n-cols (last (dtype/shape src-img))
        img-dtype (dtype/elemwise-datatype src-img)
        assign-clusters-fn (make-assign-clusters-fn n-cols img-dtype)]
    (resource/stack-resource-context
     (let [[height width channels] (dtype/shape src-img)
           [n-rows n-cols] [(* (long height) (long width)) channels]
           n-rows (long n-rows)
           n-cols (long n-cols)]
       (let [dataset (-> (dtt/reshape src-img [n-rows channels])
                         (dtt/clone :container-type :native-heap
                                    :resource-type :stack))
             {:keys [centroids scores]} (kmeans++ dataset n-quantization
                                                {:n-iters n-iters
                                                 :seed seed})
             native-centroids (ensure-native centroids)
             result-img (bufimg/new-image height width (bufimg/image-type src-img))
             result-tens (dtt/new-tensor (dtype/shape dataset)
                                         :datatype (dtype/elemwise-datatype src-img)
                                         :container-type :native-heap
                                         :resource-type :stack)]
         (assign-clusters-fn dataset native-centroids result-tens)
         (log/infof "Scores: %s\nCentroids:\n%s" scores centroids)
         (dtype/copy! result-tens result-img)
         (when dst-path
           (bufimg/save! result-img dst-path))
         {:centroids (dtype/clone centroids)
          :result result-img
          :scores scores})))))



(comment
  (def jvm-tvm (time (jvm-tvm-iterate-kmeans src-input centroids)))
  (def tvm-allinone (time (tvm-all-in-one-iterate-kmeans src-input centroids)))
  (dotimes [iter 10]
    (let [n-quantization (* (+ iter 1) 5)]
      (log/infof "Quantization: %d" n-quantization)
      (quantize-image "test/data/castle.jpg" (format "quantized-%d.png" n-quantization)
                      n-quantization)))
  )
