(ns tvm-clj.image.resize-test
  (:require [clojure.test :refer :all]
            [tech.compute.tvm.image.resize :as resize]
            [tech.compute.verify.tensor :as vf]
            [tech.compute :as compute]
            [tech.compute.tvm :as tvm]
            [tech.compute.tvm.cpu :as cpu]
            [tvm-clj.compile-test :as compile-test]
            [clojure.core.matrix :as m]
            [tech.compute.tensor :as ct]
            [tech.datatype.base :as dtype]
            [tech.resource :as resource]
            [tech.opencv :as opencv]
            [clojure.pprint :as pp]
            [clojure.core.matrix.stats :as stats]))


(defn result-tensor->opencv
  [result-tens]
  (let [[height width n-chan] (m/shape result-tens)
        out-img (opencv/new-mat height width 3 :dtype :uint8)
        out-img-tens (tvm/as-cpu-tensor out-img)]
    (ct/assign! out-img-tens result-tens)
    (compute/sync-with-host ct/*stream*)
    out-img))

(defn median
  [item-seq]
  (let [num-items (count item-seq)]
    (nth (quot num-items 2) (sort item-seq))))

(defmacro simple-time
  [& body]
  `(let [warmup# (do ~@body)
         mean-time#  (->> (repeatedly
                           10
                           #(let [start# (System/currentTimeMillis)
                                  result# (do ~@body)
                                  time-len# (- (System/currentTimeMillis) start#)]
                              time-len#))
                          stats/mean)]
     (format "%3.2fms" (double mean-time#))))


(defn downsample-img
  [& {:keys [device-type]
      :or {device-type :cpu}}]
  (vf/tensor-default-context
   (tvm/driver device-type)
   :uint8
   (let [mat (opencv/load "test/data/jen.jpg")
         img-tensor (cond-> (tvm/as-cpu-tensor mat)
                      (not= :cpu device-type)
                      (ct/clone))
         [height width n-chans] (take-last 3 (m/shape img-tensor))
         new-width 512
         ratio (/ (double new-width) width)
         new-height (long (Math/round (* (double height) ratio)))
         result (ct/new-tensor [new-height new-width n-chans] :datatype :uint8)
         downsample-fn (resize/schedule-area-reduction
                        :device-type device-type
                        :img-dtype :uint8)
         ;; Call once to take out compilation time
         _ (resize/area-reduction! img-tensor result downsample-fn)
         ds-time (simple-time
                   (resize/area-reduction! img-tensor result
                                           downsample-fn)
                   (compute/sync-with-host ct/*stream*))
         opencv-res (result-tensor->opencv result)
         reference (opencv/new-mat new-height new-width 3 :dtype :uint8)
         ref-time (simple-time
                    (opencv/resize-imgproc mat reference :linear))
         filter-fn (resize/schedule-bilinear-filter-fn
                    :device-type device-type
                    :img-dtype :uint8)
         _ (resize/bilinear-filter! img-tensor result filter-fn)
         classic-time (simple-time
                        (resize/bilinear-filter! img-tensor result filter-fn)
                        (compute/sync-with-host ct/*stream*))
         class-res (result-tensor->opencv result)
         opencv-area (opencv/new-mat new-height new-width 3 :dtype :uint8)
         area-time (simple-time
                     (opencv/resize-imgproc mat opencv-area :area))]
     (opencv/save opencv-res "tvm_area.jpg")
     (opencv/save class-res "tvm_bilinear.jpg")
     (opencv/save reference "opencv_bilinear.jpg")
     (opencv/save opencv-area "opencv_area.jpg")
     {:tvm-area ds-time
      :opencv-bilinear ref-time
      :tvm-bilinear classic-time
      :opencv-area area-time})))


(deftest resize-test
  (->> (for [dev-type [:cpu :opencl :cuda]]
         (try
           (-> (downsample-img :device-type dev-type)
               (assoc :device-type dev-type))
           (catch Throwable e nil)))
       (remove nil?)
       (pp/print-table [:device-type
                        :opencv-area :tvm-area
                        :opencv-bilinear :tvm-bilinear])))
