(ns tvm-clj.schedule
  "After describing the algorithm, the user creates a 'schedule' for the
  algorithm which involve transformations to the algorithm that are guaranteed
  not to change the results such as the tiling a computation across a tensor."
  (:require [tvm-clj.impl.protocols :refer [->node] :as bindings]
            [tvm-clj.impl.node :as jna-node]
            [tvm-clj.ast :as ast]
            [tvm-clj.impl.fns.te :as te-fns]))


(defn throw-nil
  [item key-val]
  (if-let [retval (get item key-val)]
    retval
    (throw (ex-info "Expected object but got nil"
                    {:item item
                     :key key-val}))))


(defn create-schedule
  [op-seq]
  (let [op-seq (->> (if-not (sequential? op-seq)
                      [op-seq]
                      op-seq)
                    (mapv ast/->operation))]
    (te-fns/CreateSchedule op-seq)))


(defn ->stage
  [stage-or-schedule operation]
  (case (bindings/node-type-name stage-or-schedule)
    "Stage" stage-or-schedule
    "Schedule" (throw-nil (:stage_map stage-or-schedule)
                          (ast/->operation operation))))


(defmethod jna-node/get-extended-node-value :schedule
  [node-handle item-key]
  (->stage node-handle (ast/->operation item-key)))


(defn stage-split-axis
  [stage iter-var factor]
  (te-fns/StageSplitByFactor stage iter-var factor))


(defn stage-bind
  "Bind an iter-var to a stage variable"
  [stage iter-var thread-ivar]
  (te-fns/StageBind stage iter-var thread-ivar))


(defn stage-compute-at
  "Compute src stage at dst stage dst axis"
  [src-stage dst-stage dst-axis]
  (te-fns/StageComputeAt src-stage dst-stage dst-axis))


(defn stage-fuse
  "Fuse n-axis together, returns single new axis"
  [stage axis-args]
  ;;If there is only one axis, then fusing is pointless
  (if (= 1 (count axis-args))
    (first axis-args)
    (te-fns/StageFuse stage axis-args)))


(defn stage-parallel
  "Indicate that this axis has complete parallelism"
  [stage axis]
  (te-fns/StageParallel stage axis))


(defn stage-inline
  [stage]
  (te-fns/StageComputeInline stage))


(defn stage-tile
  [stage outer-axis inner-axis outer-dim inner-dim]
  (te-fns/StageTile stage outer-axis inner-axis outer-dim inner-dim))


(defn stage-reorder
  [stage axis-seq]
  (te-fns/StageReorder stage axis-seq))


(defn stage-vectorize
  [stage axis]
  (te-fns/StageVectorize stage axis))


(defn stage-unroll
  [stage axis]
  (te-fns/StageUnroll stage axis))


(defn schedule-cache-write
  "Returns a new tensor"
  [schedule tensor cache-type]
  (let [retval (te-fns/ScheduleCacheWrite schedule tensor cache-type)]
    {:tensor retval
     :schedule schedule}))


(defn schedule-cache-read
  [schedule tensor cache-type readers]
  (throw (ex-info "Unimplemented" {})))


(defn stage-bind-gpu
  "Bind the gpu-defined axis to the tvm axis.
  GPU (cuda, opencl) define a roughly level stage breakdown of axis: block and thread.
  Threads run on the same block and can share a special kind of memory (called shared
  memory).  There can be up to 3 tvm axis per block or thread and these are labeled
  (outer iterator to inner iterator):
  [z y x]"
  [stage block-axis-seq thread-axis-seq]
  (let [axis-names ["z" "y" "x"]
        full-info-fn (fn [grp-name axis-seq]
                         (map vector
                              (repeat grp-name)
                              axis-seq
                              ;;map to axis such that if you have one, it becomes
                              ;;the x axis.  If you have 2, first is y and second
                              ;;is x, etc.
                              (drop (- 3 (count axis-seq)) axis-names)))]
    (when-not (and (<= (count block-axis-seq) 3)
                   (<= (count thread-axis-seq) 3))
      (throw (ex-info "Block, threads can have up to 3 axis"
                      {:thread-axis-count (count thread-axis-seq)
                       :block-axis-count (count block-axis-seq)})))
    (->> (concat (full-info-fn "blockIdx" block-axis-seq)
                 (full-info-fn "threadIdx" thread-axis-seq))
         (map (fn [[grp-name axis gpu-axis-name]]
                (stage-bind stage axis
                            (ast/name->thread-axis-iterator
                             (str grp-name "." gpu-axis-name)))))
         dorun)))


(defn stage-gpu-injective
  [stage op & {:keys [thread-count axis]
               :or {thread-count 16}}]

  (let [retval stage
        op (ast/->operation op)
        stage (->stage stage op)
        fused-axis (stage-fuse stage (or axis (:axis op)))
        [bx tx] (stage-split-axis stage fused-axis thread-count)]
    (stage-bind-gpu stage [bx] [tx])
    retval))


(defn stage-cpu-injective
  [stage op & {:keys [axis]}]
  (let [retval stage
        op (ast/->operation op)
        stage (->stage stage op)
        fused-axis (stage-fuse stage (or axis (:axis op)))]
    (stage-parallel stage fused-axis)
    retval))
