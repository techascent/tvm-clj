(ns tvm-clj.application.kmeans
  "TVM/JVM comparison of the kmeans algorithm components."
  (:require [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.reductions :as dtype-reduce]
            [tech.v3.tensor :as dtt]
            [tech.v3.libs.buffered-image :as bufimg]
            [tvm-clj.ast :as ast]
            [tvm-clj.impl.fns.topi :as topi-fns]
            [tvm-clj.ast.elemwise-op :as ast-op]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.compiler :as compiler]
            [tvm-clj.module :as module]
            [tvm-clj.device :as device]
            [primitive-math :as pmath]
            [tech.v3.resource :as resource]
            [clojure.tools.logging :as log])
  (:import [java.util Random List Map$Entry]
           [java.util.function Consumer]
           [tech.v3.datatype DoubleReader Buffer IndexReduction
            Consumers$StagedConsumer NDBuffer LongReader]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- inline-dis-sq
  ^double [^Buffer lhs ^Buffer rhs]
  (let [n-elems (.lsize lhs)]
    (loop [idx 0
           result 0.0]
      (if (< idx n-elems)
        (recur (unchecked-inc idx)
               (let [tmp (- (.readDouble lhs idx)
                            (.readDouble rhs idx))]
                 (pmath/+ result
                          (pmath/* tmp tmp))))
        result))))


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


(defn- choose-next-center
  ^long [random distances]
  (let [next-flt (.nextDouble ^Random random)
        indexes (dtype/->buffer (argops/argsort :tech.numerics/> distances))
        normalized-distances (dfn// distances (dfn/sum distances))
        n-rows (dtype/ecount distances)]
    (long (loop [idx 0
                 sum 0.0]
            (if (and (< idx n-rows)
                     (< sum next-flt))
              (recur (unchecked-inc idx)
                     (+ sum (double (normalized-distances
                                     (indexes idx)))))
              (indexes (dec idx)))))))


(defn choose-centers++
  "Choose centers using the kmeans++ algorithm.  Input is expected to be
  be a matrix.  Returns a smaller matrix of initial centers."
  ([dataset n-centers {:keys [seed]}]
   (let [^Random random (seed->random seed)
         [n-rows n-cols] (dtype/shape dataset)
         dataset (dtt/ensure-tensor dataset)
         ;;Create a concrete set of rows.  We will iterate over these
         ;;quite a lot.
         ^List rows (->> (dtt/rows dataset)
                         (mapv dtype/->buffer))
         n-rows (count rows)
         n-centers (long n-centers)
         initial-seed-idx (.nextInt random n-rows)
         result (dtype/make-list :int64)
         distances (dtype/make-container :float64 n-rows)]
     (.add result initial-seed-idx)
     (dotimes [iter (dec n-centers)]
       (let [centers (->> (dtt/select dataset result :all)
                          (dtt/rows)
                          (mapv dtype/->buffer))
             distances (dtype/copy! (reify DoubleReader
                                      (lsize [rdr] n-rows)
                                      (readDouble [rdr idx]
                                        (->> centers
                                             (map #(inline-dis-sq (.get rows idx) %))
                                             (apply min)
                                             (double))))
                                    distances)
             next-center-idx (choose-next-center random distances)]
         (.add result next-center-idx)))
     (dtt/select dataset result :all)))
  ([dataset n-centers]
   (choose-centers++ dataset n-centers nil)))


(defn tvm-choose++-algo
  []
  (let [n-centers (ast/variable "n_centers")
        n-rows (ast/variable "nrows")
        n-cols (ast/variable "ncols")
        ;;The distance calculation is the only real issue here.
        ;;Everything else, sort, etc. is pretty quick and sorting
        ;;isn't something available for GPU compute.
        center-indexes (ast/placeholder [n-centers] "center_indexes" :dtype :int32)
        dataset (ast/placeholder [n-rows n-cols] "dataset" :dtype :float32)
        squared-differences-op (ast/compute
                                [n-rows n-centers n-cols]
                                (ast/tvm-fn
                                 [row-idx center-idx col-idx]
                                 (ast/tvm-let
                                  [row-elem (ast/tget dataset [row-idx col-idx])
                                   center-elem (ast/tget dataset [(ast/tget center-indexes [center-idx])
                                                                  col-idx])
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
        mindistances-op (ast/compute
                         [n-rows]
                         (ast/tvm-fn
                          [row-idx]
                          (ast/commutative-reduce
                           (ast/tvm-fn->commutative-reducer
                            (ast/tvm-fn
                             [lhs rhs]
                             (ast-op/min lhs rhs))
                            [(ast-op/max-value :float32)])
                           [{:domain [0 n-centers] :name "center-idx"}]
                           [(fn [center-idx]
                              (ast/tget expanded-distances [row-idx center-idx]))]))
                         "mindistances")
        mindistances (first (ast/output-tensors mindistances-op))


        schedule (schedule/create-schedule mindistances-op)
        stage-map (:stage_map schedule)
        sq-stage (get stage-map squared-differences-op)
        ep-stage (get stage-map expanded-distances-op)
        final-stage (get stage-map mindistances-op)
        [exp-dist-row exp-dist-center] (:axis expanded-distances-op)
        [min-dist-row] (:axis mindistances-op)]

    (schedule/stage-compute-at sq-stage ep-stage exp-dist-center)
    (schedule/stage-compute-at ep-stage final-stage min-dist-row)
    (schedule/stage-parallel final-stage min-dist-row)
    {:arguments [dataset center-indexes mindistances]
     :schedule schedule}))


(defn- view-ir
  [{:keys [schedule arguments]}]
  (compiler/lower schedule arguments {:name "view_ir"}))


(defn- compile-cpu-choose++
  "Step 3 you compile it to a module, find the desired function, and
  wrap it with whatever wrapping code you need."
  []
  (let [algo (tvm-choose++-algo)
        module (compiler/compile {"cpu_distances" algo})
        low-level-fn (module/find-function module "cpu_distances")
        ref-map {:module module}]
    (fn [dataset center-indexes]
      (resource/stack-resource-context
       (let [tvm-input (if-not (dtype/as-native-buffer dataset)
                         (dtt/clone dataset
                                    :container-type :native-heap
                                    :datatype :float32
                                    :resource-type :auto)
                         dataset)
             [n-rows n-cols] (dtype/shape dataset)
             center-indexes (if-not (dtype/as-native-buffer center-indexes)
                              (dtt/clone center-indexes
                                         :container-type :native-heap
                                         :datatype :int32
                                         :resource-type :auto)
                              center-indexes)
             tvm-output (device/device-tensor [n-rows] :float32 :cpu 0)]
        ;;;Dereference ref-map
         (ref-map :module)
         (low-level-fn tvm-input center-indexes tvm-output)
         (dtt/clone tvm-output))))))


(defn tvm-choose-centers++
  ([dataset n-centers {:keys [seed]}]
   (resource/stack-resource-context
    (let [dataset (dtt/clone dataset
                             :container-type :native-heap
                             :datatype :float32
                             :resource-type :auto)
          random (seed->random seed)
          [n-rows n-cols] (dtype/shape dataset)
          center-indexes (dtt/new-tensor [n-centers]
                                         :container-type :native-heap
                                         :datatype :int32
                                         :resource-type :auto)
          initial-seed-idx (.nextInt random (int n-rows))
          _ (log/infof "first center chosen: %d" initial-seed-idx)
          _ (dtt/mset! center-indexes 0 initial-seed-idx)
          distance-fn (compile-cpu-choose++)
          n-centers (long n-centers)]
      (dotimes [idx (dec n-centers)]
        (log/infof "choosing index %d" (inc idx))
        (let [distances (distance-fn
                         dataset
                         (dtt/select center-indexes (range (inc idx))))
              _ (log/info "Distances calculated")
              next-center-idx (choose-next-center random distances)]
          (log/infof "center chosen: %d" next-center-idx)
          (dtt/mset! center-indexes (inc idx) next-center-idx)))
      (dtype/clone center-indexes))))
  ([dataset n-centers]
   (tvm-choose-centers++ dataset n-centers nil)))


(deftype KMeansReducer [n-columns
                        ^doubles data
                        ^{:unsynchronized-mutable true
                          :tag 'long} col-count]
  Consumers$StagedConsumer
  (inplaceCombine [this other] (throw (Exception. "Unimplemented")))
  (value [this] (dfn// data col-count))
  Consumer
  (accept [this row]
    (let [^Buffer row row]
      (dotimes [idx n-columns]
        (aset data idx (pmath/+ (aget data idx)
                                (.readDouble row idx))))
      (set! col-count (unchecked-inc (long col-count))))))


(defn- min-idx-reader
  ^Buffer [^List rows ^List centers ^long n-rows ^long n-centers]
  ;;Define a reader that will produce the center idx for that row
  (reify LongReader
    (lsize [rdr] n-rows)
    (readLong [rdr idx]
      (let [^NDBuffer row (.get rows idx)]
        (long
         (loop [center-idx 0
                min-distance Double/MAX_VALUE
                min-idx -1]
           (if (< center-idx n-centers)
             (let [next-dist (inline-dis-sq row (.get centers center-idx))]
               (recur (unchecked-inc center-idx)
                      (double (if (< next-dist min-distance)
                                next-dist min-distance))
                      (long (if (< next-dist min-distance)
                              center-idx min-idx))))
             min-idx)))))))


(defn kmeans-iterate
  "Iterate k-means once, returning a new matrix of centers"
  [dataset centers n-iters]
  (let [^List rows (->> (dtt/rows dataset)
                        (mapv dtype/->buffer))
        center-data (dtt/clone centers)
        ^List centers (->> (dtt/rows center-data)
                           (mapv dtype/->buffer))
        n-iters (long n-iters)
        n-rows (count rows)
        n-columns (dtype/ecount (first rows))
        n-centers (count centers)]
    (dotimes [iter n-iters]
      (log/infof "Iteration %d" iter)
      (let [min-idx-rdr (min-idx-reader rows centers n-rows n-centers)
            new-centers (->> (dtype-reduce/unordered-group-by-reduce
                              (reify IndexReduction
                                (reduceIndex [this batch-ctx ctx idx]
                                  (let [^Consumer ctx (or ctx
                                                          (KMeansReducer. n-columns
                                                                          (double-array n-columns)
                                                                          0))]
                                    (.accept ctx (.get rows idx))
                                    ctx))
                                (finalize [ths ctx]
                                  (.value ^Consumers$StagedConsumer ctx)))
                              nil
                              min-idx-rdr
                              nil)
                             (map (fn [^Map$Entry entry]
                                    [(.getKey entry) (.getValue entry)]))
                             ;;Get them in order
                             (sort-by first)
                             ;;Finalize them, getting their final values.
                             (map (comp #(.value ^Consumers$StagedConsumer %) second)))]
        (dtype/copy-raw->item! new-centers center-data)))
    center-data))


(deftype KMeansScoreReducer [center
                             ^{:unsynchronized-mutable true
                               :tag 'long} row-count
                             ^{:unsynchronized-mutable true
                               :tag 'double} current-sum]
  Consumers$StagedConsumer
  (inplaceCombine [this other] (throw (Exception. "Unimplemented")))
  (value [this] {:avg-error (double (/ (double current-sum)
                                       (double row-count)))
                 :n-rows row-count
                 :center center})
  Consumer
  (accept [this row]
    (set! current-sum (pmath/+ (double current-sum) (inline-dis-sq row center)))
    (set! row-count (unchecked-inc (long row-count)))))


(defn kmeans-score
  "Return a sum of all distance-squared's.  This number should go down :-)."
  [dataset centers]
  (log/info "setup")
  (let [^List rows (->> (dtt/rows dataset)
                        (dtype/emap dtype/->buffer :object)
                        (dtype/clone)
                        (dtype/->buffer))
        ^List centers (->> (dtt/rows centers)
                           (dtype/emap dtype/->buffer :object)
                           (dtype/clone)
                           (dtype/->buffer))
        n-rows (count rows)
        n-centers (count centers)
        min-idx-rdr (min-idx-reader rows centers n-rows n-centers)]
    (log/info "group-by")
    (->> (dtype-reduce/unordered-group-by-reduce
          (reify IndexReduction
            (reduceIndex [this batch-ctx ctx idx]
              (let [^Consumer ctx (or ctx
                                      (KMeansScoreReducer. (.get centers
                                                                 (min-idx-rdr idx))
                                                           0
                                                           0.0))]
                (.accept ctx (.get rows idx))
                ctx))
            (finalize [ths ctx]
              (.value ^Consumers$StagedConsumer ctx)))
          nil
          min-idx-rdr
          nil)
         (map (fn [^Map$Entry entry]
                [(.getKey entry) (.getValue entry)]))
         ;;Get them in order
         (sort-by first)
         ;;Finalize them, getting their final values.
         (map (comp #(.value ^Consumers$StagedConsumer %) second)))))


(defn tvm-assign-centers-algo
  []
  (let [n-rows (ast/variable "n-rows")
        n-cols (ast/variable "n-cols")
        n-centers (ast/variable "n-centers")
        dataset (ast/placeholder [n-rows n-cols] "dataset")
        centers (ast/placeholder [n-rows n-centers] "centers")
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
        mindist-stage (stage-map mindistance-op)]
    {:schedule schedule
     :arguments [dataset centers mindistance-indexes]}))


(comment
  (do
    (def src-image (bufimg/load "test/data/jen.jpg"))
    (def src-shape (dtype/shape src-image))
    ;;Make a 2d matrix out of the image.
    (def src-input (dtt/clone (dtt/reshape src-image
                                           [(* (first src-shape)
                                               (second src-shape))
                                            (last src-shape)])
                              :datatype :float32)))
  (def jvm-center-indexes (time (choose-centers++ src-input 5 {:seed 5})))
  (def tvm-center-indexes (time (tvm-choose-centers++ src-input 5 {:seed 5})))
  (def centers (dtt/select src-input tvm-center-indexes :all))

  (def original-score (kmeans-score src-input centers))
  (def centers-after-iter (kmeans-iterate src-input centers 4))
  (def after-iter-score (kmeans-score src-input centers-after-iter))


  )
