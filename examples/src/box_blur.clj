(ns box-blur
  (:require [tvm-clj.api :as api]
            [tech.opencv :as opencv]
            [tech.datatype.base :as dtype]
            [tech.compute.tvm :as tvm]
            [tech.compute.verify.tensor :as vf]
            [tech.compute :as compute]
            [clojure.core.matrix :as m]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.defaults :as ct-defaults]
            [tvm-clj.api-sugar :refer :all]
            [tech.compute.tvm.cpu]
            [tech.compute.tvm.gpu]
            [tech.compute.tvm.tensor-math])
  (:refer-clojure :exclude [+ - / * cast min max = rem])
  (:import [org.bytedeco.javacpp opencv_imgproc]))


(set! *warn-on-reflection* true)
;;Note we aren't warning on boxed math.  The small amounts of boxed
;;math that happen in this file are irrelevant for speed purposes; its
;;all compiled to the specific backend.

;;A 3x3 box blur.  Using the api sugar liberally to make the code shorter
;;and more readable.
(defn box-blur-fn
  [intermediate-datatype]
  (let [in-height (tvar "height")
        in-width (tvar "width")
        in-channels (int 3)
        input (placeholder [in-height in-width in-channels] "input"
                           :dtype :uint8)

        clamp (fn [val val_min val_max]
                (-> (min val val_max)
                    (max val_min)))
        ;;Read a clamped-to-edge value of the source image
        read-element (fn [img y x c]
                       (img [(clamp y 0 (- in-height 1))
                             (clamp x 0 (- in-width 1))
                             c]))
        padded-input (compute
                      [(+ in-height 2) (+ in-width 2) in-channels]
                      (lambda [y x c]
                              (-> (read-element input (- y 1) (- x 1) c)
                                  (cast intermediate-datatype)))
                      "padded")
        x-blur-vec (compute
                    [(+ in-height 2) in-width in-channels]
                    (lambda
                     [y x c]
                     (+ (padded-input [y x c])
                        (padded-input [y (+ x 1) c])
                        (padded-input [y (+ x 2) c])))
                    "x-blur")
        y-blur-vec (compute
                    [in-height in-width in-channels]
                    (lambda [y x c]
                            (+ (x-blur-vec [y x c])
                               (x-blur-vec [(+ y 1) x c])
                               (x-blur-vec [(+ y 2) x c])))
                    "y-blur")
        output (compute
                [in-height in-width in-channels]
                (lambda [y x c]
                        (-> (y-blur-vec [y x c])
                            (/ (const 9 intermediate-datatype))
                            (cast :uint8)))
                "box-blur")]
    {:input input
     :padded padded-input
     :x-blur x-blur-vec
     :y-blur y-blur-vec
     :box-blur output
     :output output
     :in-height in-height
     :in-width in-width
     :in-channels in-channels
     :arglist [input output]}))


(defn box-blur-no-sugar
  [intermediate-datatype]
  (let [in-height (api/variable "height")
        in-width (api/variable "width")
        in-channels (api/const 3 :int32)
        input (api/placeholder [in-height in-width in-channels] "input"
                               :dtype :uint8)

        clamp (api/tvm-fn
               [val val_min val_max]
               (->
                (api/min val val_max)
                (api/max val_min)))
        ;;Read a clamped-to-edge value of the source image
        read-element (api/tvm-fn
                      [img y x c]
                      (api/tget img [(clamp y 0 (api/sub in-height 1))
                                     (clamp x 0 (api/sub in-width 1))
                                     c]))
        padded-input-op (api/compute
                         [(api/add in-height 2) (api/add in-width 2) in-channels]
                         (api/tvm-fn
                          [y x c]
                          (-> (read-element input (api/sub y 1) (api/sub x 1) c)
                              (api/cast intermediate-datatype)))
                         "padded")
        padded-input (first (api/output-tensors padded-input-op))
        ;; We know that we will access this blur incrementing the y axis.
        ;; So we want to store the results of this computation such that
        ;; Y is the most rapidly changing (elements in Y are adjacent in memory).
        ;;
        ;; It would be idea if this could happen at the scheduling phase but
        ;; currently I do not believe it is possible.
        x-blur-op (api/compute
                   [(api/add in-height 2) in-width in-channels]
                    (api/tvm-fn
                     [y x c]
                     (api/div
                      (->
                       (api/add (api/tget padded-input [y (api/add x 2) c])
                                (api/tget padded-input [y x c]))
                       (api/add (api/tget padded-input [y (api/add x 1) c])))
                      (api/const 3 intermediate-datatype)))
                    "x-blur")
        x-blur-vec (first (api/output-tensors x-blur-op))
        y-blur-op (api/compute
                    [in-height in-width in-channels]
                    (api/tvm-fn
                     [y x c]
                     (-> (api/div (-> (api/add
                                       (api/tget x-blur-vec [y x c])
                                       (api/tget x-blur-vec [(api/add y 1) x c]))
                                      (api/add (api/tget x-blur-vec
                                                         [(api/add y 2) x c])))
                                  (api/const 3 intermediate-datatype))))
                    "y-blur")
        y-blur-vec (first (api/output-tensors y-blur-op))
        final-cast (api/compute
                    [in-height in-width in-channels]
                    (api/tvm-fn
                     [y x c]
                     (-> (api/tget y-blur-vec [y x c])
                         (api/add (float 0.5))
                         (api/cast :uint8)))
                    "box-blur")
        output (first (api/output-tensors final-cast))]
    {:input input
     :padded padded-input-op
     :x-blur x-blur-vec
     :y-blur y-blur-vec
     :box-blur final-cast
     :output output
     :in-height in-height
     :in-width in-width
     :in-channels in-channels
     :arglist [input output]}))


(defmacro simple-time
  [& body]
  `(let [report-time#  (->> (repeatedly
                             10
                             #(let [start# (System/nanoTime)
                                    result# (do ~@body)]
                                (- (System/nanoTime) start#)))
                            (map #(/ (double %)
                                     100000.0))
                            sort)]
     (apply  format "top-3 times: %.2fms %.2fms %.2fms"
             (take 3 report-time#))))


(defn time-schedule
  [schedule-fn & {:keys [device-type algorithm]
                  :or {device-type :cpu
                       ;;Could also try no-sugar
                       algorithm box-blur-fn}}]
  (let [driver (tvm/driver device-type)]
    (vf/tensor-default-context
     driver
     :uint8
     (let [src-tensor (cond-> (opencv/load "test/data/test.jpg")
                        (not= :cpu device-type)
                        ct/clone-to-device)
           [src-height src-width src-chan] (m/shape src-tensor)
           dst-img (opencv/new-mat src-height src-width src-chan
                                   :dtype :uint8)
           dst-tensor (cond-> dst-img
                        (not= :cpu device-type)
                        ct/clone-to-device)
           ;;Try changing the intermediate datatype
           {:keys [arglist schedule bind-map]} (-> (box-blur-fn :uint16)
                                                   (schedule-fn device-type))
           _ (println (api/schedule->str schedule arglist "box_blur"))
           box-blur (tvm/schedule->fn driver {:schedule schedule
                                              :arglist arglist
                                              :name :blox-blur
                                              :bind-map (or bind-map {})})]
       ;;Note that on the cpu, the opencv image itself is used with no
       ;;copy nor transformation.  TVM can operate directly on items that
       ;;implement enough protocols:
       ;; (tech.compute.tvm.driver/acceptible-tvm-device-buffer)

        ;;warmup
        (box-blur src-tensor dst-tensor)

        _ (when-not (= :cpu device-type)
            (ct/assign! dst-img dst-tensor)
            (compute/sync-with-host (ct-defaults/infer-stream {})))
        (let [time-result
              (simple-time
               (box-blur src-tensor dst-tensor)
               (compute/sync-with-host (ct-defaults/infer-stream {})))]

          (opencv/save dst-img (format "result-%s.jpg" (name device-type)))
          time-result)))))


(defn base-schedule
  [item device-type]
  (let [target (:box-blur item)
        schedule (api/create-schedule [target])]
    (assoc item :schedule schedule
           :arglist [(:input item) target]
           :output target)))


(defn parallel
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-parallel (schedule box-blur) y-axis)
    (assoc item :schedule schedule)))


(defn reorder
  [{:keys [box-blur y-blur x-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-x-axis x-chan-axis x-y-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-reorder (schedule x-blur) [x-y-axis x-x-axis x-chan-axis])
    (assoc item :schedule schedule)))


(defn tiled-schedule
  [{:keys [box-blur y-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis y-blur)
        tile-axis
        (api/stage-tile (schedule y-blur)
                        y-axis x-axis 16 3)]
    (assoc item
           :schedule schedule
           :tile-axis tile-axis)))



(defn inline
  [{:keys [padded x-blur y-blur box-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])]
    (api/stage-inline (schedule padded))
    (api/stage-inline (schedule x-blur))
    (api/stage-inline (schedule y-blur))
    (api/stage-parallel (schedule box-blur)
                        (api/stage-fuse (schedule box-blur)
                                        (:axis box-blur)))
    (assoc item :schedule schedule)))


(defn compute-at
  [{:keys [padded x-blur y-blur box-blur] :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [y-axis x-axis chan-axis] (:axis box-blur)]
    (api/stage-inline (schedule x-blur))
    (api/stage-inline (schedule padded))
    (api/stage-compute-at (schedule y-blur)
                          (schedule box-blur) y-axis)
    (assoc item :schedule schedule)))


(defn all-the-toys
  [{:keys [padded box-blur x-blur y-blur in-height in-width in-channels]
    :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-blur-y-axis x-blur-x-axis x-blur-chan-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)
        [y-blur-y-axis y-blur-x-axis y-blur-chan-axis] (:axis y-blur)
        [y-outer x-outer y-inner x-inner] (api/stage-tile
                                           (schedule box-blur)
                                           y-axis x-axis
                                           8 8)
        ;;Here I had to ask some questions:
        ;;
        ;;https://discuss.tvm.ai/t/how-to-control-data-layout-of-intermediate-values/898/4

        ;;We want to calculate X in xyc order but store in xcy order as that is the
        ;;order it is accessed.  This requires setting up a cache operation.  First,
        ;;reorder X so that the computation happens in xcy order.  Note that this will
        ;;make calculating X inefficient at this point due to the extremely out-of-order
        ;;access of the padded input:.  It will make the y-blur more efficient, however.
        _ (api/stage-reorder (schedule x-blur) [x-blur-x-axis x-blur-chan-axis
                                                x-blur-y-axis])
        {x-blur-cache :tensor
         schedule :schedule} (api/schedule-cache-write schedule x-blur "local")
        [cache-x-axis cache-chan-axis cache-y-axis] (:axis x-blur-cache)]
    (api/stage-inline (schedule padded))
    (api/stage-inline (schedule x-blur))
    (api/stage-inline (schedule y-blur))
    ;;Reorder the cache stage to compute in y,x,c order.  This means we will read input
    ;;in a sane order but write to the cache out of order.  This is fine because we
    ;;aren't using those writes for a while meaning the memory stalls will be ignored
    ;;till much later.  Since we are writing thread-local only the system may not even
    ;;write the cache back out to main memory saving a ton of time.
    (api/stage-reorder (schedule x-blur-cache) [cache-y-axis cache-x-axis
                                                cache-chan-axis])
    ;;If we travel down the y dimension in the tile then we do not have to precompute
    ;;as much
    (if (= device-type :cpu)
      (do
        ;;schedule the cache operation to happen
        (api/stage-compute-at (schedule x-blur-cache)
                              (schedule box-blur) y-inner)
        (api/stage-parallel (schedule box-blur) y-outer))
      (let [gpu-thread-axis (api/stage-fuse (schedule box-blur)
                                            [x-inner chan-axis])]
        (api/stage-compute-at (schedule x-blur-cache)
                              (schedule box-blur) gpu-thread-axis)
        ;;We lose the cache here really but it does at least run and produce a correct
        ;;result.  Really, on a per-block basis, you would want to build the cache as a
        ;;first step using all the threads but that leads to more sophistication than I
        ;;wanted in a demo.  For homework, go through the gpu conv layer tvm tutorial
        ;;and then apply it here.
        (api/stage-bind-gpu (schedule box-blur)
                            [(api/stage-fuse (schedule box-blur)
                                             [y-outer x-outer y-inner])]
                            [gpu-thread-axis])))
    (assoc item :schedule schedule)))


(defn all-the-toys-vertical-thread-cpu
  [{:keys [padded box-blur x-blur y-blur in-height in-width in-channels]
    :as item} device-type]
  (let [schedule (api/create-schedule [box-blur])
        [x-blur-y-axis x-blur-x-axis x-blur-chan-axis] (:axis x-blur)
        [y-axis x-axis chan-axis] (:axis box-blur)
        [y-blur-y-axis y-blur-x-axis y-blur-chan-axis] (:axis y-blur)
        ;;We want computations to run vertically down the image
        [x-outer x-inner] (api/stage-split-axis (schedule box-blur) x-axis 16)
        _ (api/stage-reorder (schedule x-blur) [x-blur-x-axis x-blur-chan-axis x-blur-y-axis])
        {x-blur-cache :tensor
         schedule :schedule} (api/schedule-cache-write schedule x-blur "local")
        [cache-x-axis cache-chan-axis cache-y-axis] (:axis x-blur-cache)]
    (api/stage-inline (schedule padded)) ;;inline into x-blur
    (api/stage-inline (schedule x-blur)) ;;inline into x-blur-cache
    (api/stage-inline (schedule y-blur)) ;;inline into box-blur
    ;;Reorder the cache stage to compute in y,x,c order.  This means we will read input
    ;;in a sane order but write to the cache out of order.  This is fine because we
    ;;aren't using those writes for a while meaning the memory stalls will be ignored
    ;;till much later.  Since we are writing thread-local only the system may not even
    ;;write the cache back out to main memory saving a ton of time.
    (api/stage-reorder (schedule x-blur-cache) [cache-y-axis cache-x-axis
                                                cache-chan-axis])
    (api/stage-reorder (schedule box-blur) [x-outer y-axis x-inner chan-axis])
    (if (= device-type :cpu)
      (do
        ;;schedule the cache operation to happen
        (api/stage-compute-at (schedule x-blur-cache)
                              (schedule box-blur) y-axis)
        (api/stage-parallel (schedule box-blur) x-outer)
        (api/stage-unroll (schedule x-blur-cache) cache-chan-axis)
        (api/stage-unroll (schedule box-blur) chan-axis))
      (throw (ex-info "GPU version not implemented" {})))
    (assoc item :schedule schedule)))


(defn time-opencv
  []
  (let [src-img (opencv/load "test/data/test.jpg")
        [src-height src-width src-chan] (m/shape src-img)
        dst-img (opencv/new-mat src-height src-width src-chan
                                :dtype :uint8)]
    ;;warmup, load the images into cache
    (opencv_imgproc/blur src-img dst-img (opencv/size 3 3))
    (let [time-result
          (simple-time
            (opencv_imgproc/blur src-img dst-img (opencv/size 3 3)))]
          (opencv/save dst-img "reference.jpg")
          time-result)))
