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
            [tvm-clj.ast.elemwise-op :as ast-op]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tvm-clj.module :as module]
            [tvm-clj.device :as device]
            [tvm-clj.impl.fns.tvm.contrib.sort :as contrib-sort]
            [primitive-math :as pmath]
            [tech.v3.resource :as resource]
            [clojure.tools.logging :as log])
  (:import [java.util Random List Map$Entry]
           [java.util.function Consumer LongConsumer]
           [smile.clustering KMeans]
           [tech.v3.datatype DoubleReader Buffer IndexReduction
            Consumers$StagedConsumer NDBuffer LongReader
            ArrayHelpers]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


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


(defn- ensure-native
  ^NDBuffer [tensor]
  (if-not (dtype/as-native-buffer tensor)
    (dtt/clone tensor
               :container-type :native-heap
               :datatype :float32
               :resource-type :auto)
    tensor))


(defn- choose-next-center
  ^long [random distances]
  (let [next-flt (.nextDouble ^Random random)
        ;;No one (not intel, not smile) actually sorts the distances
        ;;_ (contrib-sort/argsort distances indexes 0 false)
        distance-sum (dfn/sum distances)
        n-rows (dtype/ecount distances)
        target-amt (* next-flt distance-sum)
        ^NDBuffer distances distances
        retval
        (long (loop [idx 0
                     sum 0.0]
                (if (and (< idx n-rows)
                         (< sum target-amt))
                  (recur (unchecked-inc idx)
                         (+ sum (double (.ndReadDouble distances idx))))
                  (do
                    (log/infof "Finished - rel-linear-idx %d - %f" idx (/ (double idx)
                                                                          (double n-rows)))
                    idx))))]
    retval))


(defn choose-centers++
  "Implementation of the kmeans++ center choosing algorithm.  Distance-fn takes
  three arguments: dataset, centers, and distances and must mutably write
  it's result into distances."
  [dataset n-centers distance-fn {:keys [seed]}]
  (resource/stack-resource-context
   (let [dataset (ensure-native dataset)
         random (seed->random seed)
         [n-rows n-cols] (dtype/shape dataset)
         centers (dtt/new-tensor [n-centers n-cols]
                                 :container-type :native-heap
                                 :datatype :float32
                                 :resource-type :auto)
         distances (dtt/new-tensor [n-rows]
                                   :container-type :native-heap
                                   :datatype :float32
                                   :resource-type :auto)
         initial-seed-idx (.nextInt random (int n-rows))
         _ (log/infof "first center chosen: %d" initial-seed-idx)
         _ (dtt/mset! centers 0 (dtt/mget dataset initial-seed-idx))
         n-centers (long n-centers)]
     (dotimes [idx (dec n-centers)]
       (log/infof "choosing index %d" (inc idx))
       (distance-fn dataset centers idx distances)
       (log/infof "Distances calculated: %s" (vec (take 5 distances)))
       (let [next-center-idx (choose-next-center random distances)]
         (log/infof "center chosen: %d" next-center-idx)
         (dtt/mset! centers (inc idx) (dtt/mget dataset next-center-idx))))
     ;;copy centers back into JVM land.
     (dtype/clone centers))))


(defn- group-by-centers
  "Group by centers, aggregating into a reducer and returning
  a sequence of center-idx, reduced value."
  [dataset centers ^NDBuffer center-indexes assign-centers-fn reducer-fn]
  ;;Potentially centers have already been assigned
  (when assign-centers-fn
    (log/info "Assigning centers")
    (assign-centers-fn dataset centers center-indexes))
  (log/info "Grouping by center")
  (->> (dtype-reduce/unordered-group-by-reduce
        (reify IndexReduction
          (reduceIndex [this batch-ctx ctx row-idx]
            (let [^LongConsumer ctx (or ctx (reducer-fn (.ndReadLong center-indexes row-idx)))]
              (.accept ctx row-idx)
              ctx)))
        nil
        center-indexes
        nil)
       (map (fn [^Map$Entry entry]
              [(.getKey entry)
               (.value ^Consumers$StagedConsumer (.getValue entry))]))
       (sort-by first)
       (map second)))


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


(deftype KMeansScoreReducer [^NDBuffer dataset
                             ^NDBuffer centers
                             center-idx
                             ^long n-cols
                             ^{:unsynchronized-mutable true
                               :tag 'long} n-rows
                             ^{:unsynchronized-mutable true
                               :tag 'double} current-sum]
  Consumers$StagedConsumer
  (inplaceCombine [this other] (throw (Exception. "Unimplemented")))
  (value [this] {:avg-error (double (/ (double current-sum)
                                       (double n-rows)))
                 :n-rows n-rows
                 :center (dtt/select centers center-idx :all)})
  LongConsumer
  (accept [this row-idx]
    (set! current-sum (pmath/+ (double current-sum)
                               (row-center-distance dataset centers row-idx center-idx n-cols)))
    (set! n-rows (unchecked-inc (long n-rows)))))


(defn- make-score-reducer
  [dataset centers n-cols center-idx]
  (KMeansScoreReducer. dataset centers center-idx
                       n-cols 0 0.0))


(deftype NewCentroidReducer [^long n-cols
                             ^NDBuffer dataset
                             ^doubles data
                             ^{:unsynchronized-mutable true
                               :tag 'long} n-rows]
  LongConsumer
  (accept [this row-idx]
    (dotimes [col-idx n-cols]
      (ArrayHelpers/aset data col-idx
                         (pmath/+ (aget data col-idx)
                                  (.ndReadDouble dataset row-idx col-idx))))
    ;;Dunny why the type hinting is failing here.
    (set! n-rows (unchecked-inc (long n-rows))))
  Consumers$StagedConsumer
  (inplaceCombine [this other] (throw (Exception. "Unimplemented")))
  (value [this] (println "undiv-data" (vec data))
    (dfn// data n-rows)))


(defn- make-centroid-reducer
  [dataset n-cols center-idx]
  (NewCentroidReducer. n-cols
                       dataset
                       (double-array n-cols)
                       0))


(deftype CombinedReducer [^LongConsumer red-1
                          ^LongConsumer red-2]
  LongConsumer
  (accept [this row-idx]
    (.accept red-1 row-idx)
    (.accept red-2 row-idx))
  Consumers$StagedConsumer
  (inplaceCombine [this other] (throw (Exception. "Unimplemented")))
  (value [this] [(.value ^Consumers$StagedConsumer red-1)
                 (.value ^Consumers$StagedConsumer red-2)]))


(defn kmeans-score
  "Return a sequence of maps, one for each center, that contains keys
  `#{:avg-error :n-rows :center}`"
  ^double [dataset ^NDBuffer centers center-indexes assign-centers-fn]
  (resource/stack-resource-context
   (let [dataset (ensure-native dataset)
         [n-rows n-cols] (.shape dataset)
         [n-centers n-cols] (.shape centers)
         n-rows (long n-rows)
         n-cols (long n-cols)
         n-center (long n-centers)
         center-indexes (or center-indexes
                            (dtt/new-tensor [n-rows]
                                            :datatype :int32
                                            :container-type :native-heap
                                            :resource-type :auto))]
     (->> (group-by-centers dataset centers center-indexes assign-centers-fn
                            (partial make-score-reducer dataset centers n-cols))
          (map :avg-error)
          (apply +)
          (double)))))


(defn kmeans-iterate
  "Iterate k-means once, returning a new matrix of centers"
  [dataset centers assign-centers-fn n-iters]
  (resource/stack-resource-context
   (let [dataset (ensure-native dataset)
         ^NDBuffer centers (dtt/clone centers
                                      :datatype :float32
                                      :container-type :native-heap
                                      :resource-type :auto)
         [n-rows n-cols] (.shape dataset)
         [n-centers n-cols] (.shape centers)
         n-rows (long n-rows)
         n-cols (long n-cols)
         n-center (long n-centers)
         center-indexes (dtt/new-tensor [n-rows]
                                        :datatype :int32
                                        :container-type :native-heap
                                        :resource-type :auto)
         ;;side-effecting mapv
         scores (mapv (fn [iter-idx]
                        (log/infof "Iteration %d" iter-idx)
                        (let [combined-data (group-by-centers
                                             dataset centers center-indexes
                                             assign-centers-fn
                                             (fn [^long center-idx]
                                               (CombinedReducer. (make-score-reducer dataset centers n-cols center-idx)
                                                                 (make-centroid-reducer dataset n-cols center-idx))))]
                          ;;side effecting update centers
                          (->> (map second combined-data)
                               ;;Coalesce into a new buffer
                               (dtype/coalesce! centers))
                          (->> (map (comp :avg-error first) combined-data)
                               (apply +)
                               (double))))
                      (range n-iters))]
     (log/info "finished")
     {:centers (dtt/clone centers)
      :scores (vec (concat scores
                           [(kmeans-score dataset centers
                                          center-indexes
                                          assign-centers-fn)]))})))


(defn- jvm-distance-fn
  [^NDBuffer dataset ^NDBuffer centers center-idx ^NDBuffer distances]
  (let [[n-rows n-cols] (.shape dataset)
        [n-centers n-cols] (.shape centers)
        n-rows (long n-rows)
        n-cols (long n-cols)
        n-centers (long n-centers)
        center-idx (long center-idx)]
    (if (== center-idx 0)
      (pfor/parallel-for
       row-idx
       n-rows
       (let [new-distance (row-center-distance dataset centers row-idx
                                               center-idx n-cols)]
         (.ndWriteDouble distances row-idx new-distance)))
      (pfor/parallel-for
       row-idx
       n-rows
       (let [new-distance (row-center-distance dataset centers row-idx
                                               center-idx n-cols)
             old-distance (.ndReadDouble distances row-idx)]
         (when (< new-distance old-distance)
           (.ndWriteDouble distances row-idx new-distance)))))
    distances))


(defn tvm-distances-algo
  []
  (let [n-centers (ast/variable "n_centers")
        n-rows (ast/variable "nrows")
        n-cols (ast/variable "ncols")
        center-idx (ast/variable "center-idx")
        ;;The distance calculation is the only real issue here.
        ;;Everything else, sort, etc. is pretty quick and sorting
        ;;isn't something available for GPU compute.
        centers (ast/placeholder [n-centers n-cols] "centers" :dtype :float32)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :float32)
        distances (ast/placeholder [n-rows] "distances" :dtype :float32)
        squared-differences-op (ast/compute
                                [n-rows n-cols]
                                (ast/tvm-fn
                                 [row-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (ast/tget dataset [row-idx col-idx])
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
                                  [(float 0.0)])
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
                                                     (ast-op/max-value :float32)
                                                     (ast/tget distances [row-idx]))
                            cur-dist (ast/tget expanded-distances [row-idx])]
                           (ast-op/select (ast-op/<= cur-dist prev-dist)
                                          cur-dist prev-dist)))
                         "mindistances")
        mindistances (first (ast/output-tensors mindistances-op))


        schedule (schedule/create-schedule mindistances-op)
        stage-map (:stage_map schedule)
        sq-stage (get stage-map squared-differences-op)
        ep-stage (get stage-map expanded-distances-op)
        final-stage (get stage-map mindistances-op)
        [exp-dist-row] (:axis expanded-distances-op)
        [min-dist-row] (:axis mindistances-op)]
    (schedule/stage-compute-at sq-stage ep-stage exp-dist-row)
    (schedule/stage-compute-at ep-stage final-stage min-dist-row)
    (schedule/stage-parallel final-stage min-dist-row)
    {:arguments [dataset centers center-idx distances mindistances]
     :schedule schedule}))


(defn- view-ir
  [{:keys [schedule arguments]}]
  (compiler/lower schedule arguments {:name "view_ir"}))


(defn- make-tvm-distance-fn
  "Step 3 you compile it to a module, find the desired function, and
  wrap it with whatever wrapping code you need."
  []
  (let [algo (tvm-distances-algo)
        module (compiler/compile {"cpu_distances" algo})
        low-level-fn (module/find-function module "cpu_distances")
        ref-map {:module module}]
    (fn [dataset centers center-idx distances]
      ;;;Dereference ref-map
      (ref-map :module)
      (low-level-fn dataset centers center-idx distances distances))))


(defn jvm-assign-centers
  "Given a dataset and centers, assign a center by index to each row."
  [^NDBuffer dataset ^NDBuffer centers ^NDBuffer center-indexes]
  (let [[n-rows n-cols] (.shape dataset)
        [n-centers n-cols] (.shape centers)
        n-rows (long n-rows)
        n-cols (long n-cols)
        n-centers (long n-centers)]
    (pfor/parallel-for
     row-idx
     n-rows
     (loop [center-idx 0
            min-distance Double/MAX_VALUE
            min-idx -1]
       (if (< center-idx n-centers)
         (let [new-distance (row-center-distance dataset centers row-idx
                                                 center-idx n-cols)]
           (recur (unchecked-inc center-idx)
                  (pmath/double (if (< new-distance min-distance)
                                  new-distance min-distance))
                  (pmath/long (if (< new-distance min-distance)
                                center-idx min-idx))))
         (.ndWriteLong center-indexes row-idx min-idx))))))


(defn tvm-assign-centers-algo
  []
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast/variable "n-cols")
        n-centers (ast/variable "n-centers")
        dataset (ast/placeholder [n-rows n-cols] "dataset")
        centers (ast/placeholder [n-centers n-cols] "centers")
        squared-differences-op (ast/compute
                                [n-rows n-centers n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (ast/tget dataset [row-idx col-idx])
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
                                  [(float 0.0)])
                                 [{:domain [0 n-cols] :name "col-idx"}]
                                 [(fn [col-idx]
                                    (ast/tget squared-diff [row-idx center-idx col-idx]))]))
                               "expanded-distances")
        expanded-distances (first (ast/output-tensors expanded-distances-op))
        mindistance-indexes (topi-fns/argmin expanded-distances -1 false)
        mindistance-op (:op mindistance-indexes)
        schedule (schedule/create-schedule mindistance-op)
        stage-map (:stage_map schedule)
        sq-diff-stage (stage-map squared-differences-op)
        exp-dist-stage (stage-map expanded-distances-op)
        [exp-dist-rows exp-dist-cent] (:axis expanded-distances-op)
        ;;The argmin fn uses two stages, one to that uses a temporary float,idx structure
        ;;and one that copies the indexes alone back out
        [mindist-calc-stage mindist-copy-stage] (take-last 2 (:stages schedule))
        [mindist-calc-rows] (get-in mindist-calc-stage [:op :axis])
        [mindist-copy-rows] (get-in mindist-copy-stage [:op :axis])]
    ;; (schedule/stage-compute-at sq-diff-stage exp-dist-stage exp-dist-cent)
    (schedule/stage-compute-at sq-diff-stage exp-dist-stage exp-dist-cent)
    (schedule/stage-compute-at exp-dist-stage mindist-calc-stage mindist-calc-rows)
    (schedule/stage-compute-at mindist-calc-stage mindist-copy-stage mindist-copy-rows)
    (schedule/stage-parallel mindist-copy-stage mindist-copy-rows)
    {:schedule schedule
     :arguments [dataset centers mindistance-indexes]}))


(defn make-tvm-assign-centers
  []
  (let [algo (tvm-assign-centers-algo)
        module (compiler/compile {"cpu_assign_centers" algo})
        low-level-fn (module/find-function module "cpu_assign_centers")
        ref-map {:module module}]
    (fn [dataset centers center-indexes]
      ;;;Dereference ref-map
      (ref-map :module)
      (low-level-fn dataset centers center-indexes))))


(defn tvm-aggregate-centers-algo
  "brute force aggregate centers.  Each thread gets one centroid element to agg into."
  [n-cols]
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast-op/const n-cols :int32)
        n-centers (ast/variable "n-centers")
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :float32)
        centers (ast/placeholder [n-centers n-cols] :centers :dtype :float64)
        squared-differences-op (ast/compute
                                [n-rows n-centers n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (ast/tget dataset [row-idx col-idx])
                                   center-elem (ast-op/cast (ast/tget centers [center-idx col-idx]) :float32)
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
                                  [(float 0.0)])
                                 [{:domain [0 n-cols] :name "col-idx"}]
                                 [(fn [col-idx]
                                    (ast/tget squared-diff [row-idx center-idx col-idx]))]))
                               "expanded-distances")
        expanded-distances (first (ast/output-tensors expanded-distances-op))
        center-indexes (topi-fns/argmin expanded-distances -1 false)
        mindistance-op (:op center-indexes)
        agg-op (ast/compute
                [n-centers n-cols]
                (ast/tvm-fn
                 [center-idx col-idx]
                 (ast/commutative-reduce
                  (ast/tvm-fn->commutative-reducer
                   (ast/tvm-fn
                    [fsum lsum sq-elem inc-amt]
                    [(ast-op/+ fsum sq-elem)
                     (ast-op/+ lsum inc-amt)])
                   [(double 0.0)
                    (int 0)])
                  [{:domain [0 n-rows] :name "row-idx"}]
                  [(fn [row-idx]
                     [(ast-op/select (ast-op/eq center-idx
                                                (ast/tget center-indexes [row-idx]))
                                     (ast-op/cast (ast/tget dataset [row-idx col-idx])
                                                  :float64)
                                     (double 0.0))
                      (ast-op/select (ast-op/and
                                      (ast-op/eq center-idx
                                                 (ast/tget center-indexes [row-idx]))
                                      (ast-op/eq col-idx (int 0)))
                                     (ast-op/const 1 :int32)
                                     (ast-op/const 0 :int32))])]))
                "new-centers-sum")
        new-centers (first (ast/output-tensors agg-op))
        new-counts (second (ast/output-tensors agg-op))
        div-op (ast/compute
                [n-centers n-cols]
                (ast/tvm-fn
                 [center-idx col-idx]
                 (ast-op// (ast/tget new-centers [center-idx col-idx])
                           (-> (ast/tget new-counts [center-idx 0])
                               (ast-op/cast :float64))))
                "new-centers")


        output (first (ast/output-tensors div-op))
        schedule (schedule/create-schedule div-op)
        stage-map (:stage_map schedule)
        sq-diff-stage (stage-map squared-differences-op)
        exp-dist-stage (stage-map expanded-distances-op)
        mindist-copy-stage (stage-map mindistance-op)
        mindist-calc-op (:op (first (ast/input-tensors mindistance-op)))
        mindist-calc-stage (stage-map mindist-calc-op)
        [mindist-calc-rows] (get-in mindist-calc-stage [:op :axis])
        [mindist-copy-rows] (get-in mindist-copy-stage [:op :axis])
        [exp-dist-rows exp-dist-cent] (:axis expanded-distances-op)
        agg-stage (get stage-map agg-op)
        div-stage (get stage-map div-op)
        [center-axis col-axis] (:axis div-op)]
    (schedule/stage-compute-at sq-diff-stage exp-dist-stage exp-dist-cent)
    (schedule/stage-compute-at exp-dist-stage mindist-calc-stage mindist-calc-rows)
    (schedule/stage-compute-at mindist-calc-stage mindist-copy-stage mindist-copy-rows)
    (schedule/stage-vectorize mindist-copy-stage mindist-copy-rows)
    (schedule/stage-compute-at agg-stage div-stage col-axis)

    ;;Two parallelized passes over the data per iteration


    (schedule/stage-parallel mindist-copy-stage mindist-copy-rows)

    (schedule/stage-parallel div-stage center-axis)

    {:arguments [dataset centers new-counts output]
     :mindistance-op mindistance-op
     :schedule schedule}))


(defn make-tvm-agg-centers-fn
  [n-cols]
  (let [algo (tvm-aggregate-centers-algo n-cols)
        module (compiler/compile {"cpu_agg_centers" algo})
        low-level-fn (module/find-function module "cpu_agg_centers")
        ref-map {:module module}]
    (fn [dataset center-indexes new-counts new-centers]
      ;;;Dereference ref-map
      (ref-map :module)
      (low-level-fn dataset center-indexes new-counts new-centers))))


(comment
  (do
    (def src-image (bufimg/load "test/data/jen.jpg"))
    (def src-shape (dtype/shape src-image))
    ;;Make a 2d matrix out of the image.
    (def src-input (dtt/clone (dtt/reshape src-image
                                           [(* (first src-shape)
                                               (second src-shape))
                                            (last src-shape)])
                              :datatype :float32
                              :container-type :native-heap)))
  (def jvm-centers (time (choose-centers++ src-input 5 jvm-distance-fn {:seed 5})))

  (def tvm-distance-fn (make-tvm-distance-fn))
  (def tvm-centers (time (choose-centers++ src-input 5 tvm-distance-fn {:seed 5})))

  (def tvm-assign-centers (make-tvm-assign-centers))

  (def jvm-centers-and-scores (time (kmeans-iterate src-input jvm-centers jvm-assign-centers 4)))
  ;;102 seconds
  (def tvm-centers-and-scores (time (kmeans-iterate src-input tvm-centers tvm-assign-centers 1)))

  (def jen-indexes (dtt/new-tensor [(* (first src-shape) (second src-shape))]
                                   :container-type :native-heap
                                   :datatype :int32))

  (tvm-assign-centers src-input tvm-centers (dtt/ensure-tensor jen-indexes))

  (def tvm-test-centers (dtt/clone tvm-centers
                                   :container-type :native-heap
                                   :datatype :float64))

  (def new-centers (dtt/new-tensor (dtype/shape tvm-test-centers)
                              :container-type :native-heap
                              :datatype :float64))

  (def new-counts (dtt/new-tensor (dtype/shape tvm-centers)
                                  :container-type :native-heap
                                  :datatype :int32))


  (def agg-fn (make-tvm-agg-centers-fn (last (dtype/shape src-image))))


  (time (agg-fn src-input tvm-test-centers new-counts new-centers))


  (def bad-centers (dtt/->tensor [[0 0 0]
                                  [1 0 0]
                                  [2 0 0]
                                  [3 0 0]
                                  [4 0 0]]
                                 :datatype :float32))

  (def bad-centers-and-scores (kmeans-iterate src-input bad-centers jvm-assign-centers 4))


  (def castle-img (bufimg/load "test/data/castle.jpg"))

  (def castle-shape (dtype/shape castle-img))
  (def castle-tens (-> (dtt/reshape castle-img [(* (first castle-shape)
                                                   (second castle-shape))
                                                (last castle-shape)])
                       (dtt/clone :datatype :float32)))
  (def castle-centers (choose-centers++ castle-tens 5 jvm-distance-fn {:seed 5}))
  (def jvm-castle-centers (time (kmeans-iterate castle-tens castle-centers jvm-assign-centers 4)))

  (def n-rows (first (dtype/shape castle-tens)))
  (def centers (int-array n-rows))
  (def center-indexes (jvm-assign-centers castle-tens (:centers jvm-castle-centers) (dtt/ensure-tensor centers)))
  (def dst-image (bufimg/new-image (first castle-shape) (second castle-shape) (bufimg/image-type castle-img)))
  (def dst-tens (dtt/reshape dst-image [(* (first castle-shape)
                                           (second castle-shape))
                                        (last castle-shape)]))
  (def center-tens (dtt/select (:centers jvm-castle-centers)
                               centers
                               :all))

  (dtype/copy! center-tens dst-tens)

  (bufimg/save! dst-image "castle-centers.png")

  (def jen-dst (bufimg/new-image (first src-shape) (second src-shape) (bufimg/image-type src-image)))
  (def jen-dst-tens (dtt/reshape jen-dst
                                 [(* (first src-shape)
                                     (second src-shape))
                                  (last src-shape)]))

  (def jen-indexes (dtt/new-tensor [(* (first src-shape) (second src-shape))]
                                   :container-type :native-heap
                                   :datatype :int32))

  (jvm-assign-centers src-input (:centers tvm-centers-and-scores) (dtt/ensure-tensor jen-indexes))

  (def jen-centers (dtt/select (:centers tvm-centers-and-scores)
                               jen-indexes :all))
  (dtype/copy! jen-centers jen-dst)
  (bufimg/save! jen-dst "jen-centers.png")
  )
