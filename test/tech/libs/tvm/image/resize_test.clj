(ns tech.libs.tvm.image.resize-test
  (:require [clojure.test :refer :all]
            [tech.libs.tvm.image.resize :as resize]
            [tech.compute.verify.tensor :as vf]
            [tech.compute :as compute]
            [tech.libs.tvm :as tvm]
            [tech.libs.tvm.cpu]
            [tech.libs.tvm.gpu]
            [tech.compute.tensor :as ct]
            [tech.v2.datatype :as dtype]
            [tech.v2.datatype.functional :as dfn]
            [tech.opencv :as opencv]
            [clojure.tools.logging :as log]
            [clojure.pprint :as pp]))


(defn result-tensor->opencv
  [result-tens]
  (let [[height width n-chan] (dtype/shape result-tens)
        out-img (opencv/new-mat height width 3 :dtype :uint8)]
    (ct/assign! out-img result-tens {:sync? true})
    out-img))

(defn median
  [item-seq]
  (let [num-items (count item-seq)]
    (nth (quot num-items 2) (sort item-seq))))

(defmacro simple-time
  [& body]
  `(let [warmup# (do ~@body)
         mean-time#  (-> (repeatedly
                          10
                          #(let [start# (System/currentTimeMillis)
                                 result# (do ~@body)
                                 time-len# (- (System/currentTimeMillis) start#)]
                             time-len#))
                         (dfn/descriptive-stats [:mean :min :max]))]
     (->> mean-time#
          (map (fn [[k# v#]]
                 [k# (format "%3.2fms" (double v#))]))
          (into {}))))


(defn downsample-img
  [& {:keys [device-type]
      :or {device-type :cpu}}]
  (vf/verify-context
   (tvm/driver device-type)
   :uint8
   (try
     (let [mat (opencv/load "test/data/jen.jpg")
           img-tensor (ct/ensure-device mat)
           [height width n-chans] (take-last 3 (dtype/shape img-tensor))
           new-width 512
           ratio (/ (double new-width) width)
           new-height (long (Math/round (* (double height) ratio)))
           result (ct/new-tensor [new-height new-width n-chans])
           downsample-fn (resize/schedule-area-reduction
                          :device-type device-type
                          :img-dtype :uint8)
           ;; Call once to take out compilation time
           _ (resize/area-reduction! img-tensor result downsample-fn)
           ds-time (simple-time
                    (resize/area-reduction! img-tensor result downsample-fn)
                    (compute/sync-with-host))
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
                         (compute/sync-with-host))
           class-res (result-tensor->opencv result)
           opencv-area (opencv/new-mat new-height new-width 3 :dtype :uint8)
           area-time (simple-time
                      (opencv/resize-imgproc mat opencv-area :area))]
       (opencv/save opencv-res "tvm_area.jpg")
       (opencv/save class-res "tvm_bilinear.jpg")
       (opencv/save reference "opencv_bilinear.jpg")
       (opencv/save opencv-area "opencv_area.jpg")
       [(assoc ds-time :device-type device-type :resize-op :tvm-area)
        (assoc ref-time :device-type device-type :resize-op :opencv-bilinear)
        (assoc classic-time :device-type device-type :resize-op :tvm-bilinear)
        (assoc area-time :device-type device-type :resize-op :opencv-area)])
     (catch Throwable e
       (log/errorf e "Failure attempting to run device %s" device-type)))))


(deftest resize-test
  (->> (for [dev-type [:cpu :opencl :cuda]]
         (try
           (downsample-img :device-type dev-type)
           (catch Throwable e nil)))
       (remove nil?)
       (apply concat)
       (pp/print-table)))
