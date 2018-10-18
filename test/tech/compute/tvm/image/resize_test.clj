(ns tvm-clj.image.resize-test
  (:require [clojure.test :refer :all]
            [tvm-clj.image.resize :as resize]
            [tech.compute.verify.tensor :as vf]
            [tvm-clj.compute.registry :as registry]
            [tvm-clj.compute.cpu :as cpu]
            [tvm-clj.compile-test :as compile-test]
            [clojure.core.matrix :as m]
            [tech.compute.tensor :as ct]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [tech.resource :as resource]
            [tech.opencv :as opencv]
            [tvm-clj.compute.tensor-math :as tvm-tm]
            [clojure.pprint :as pp]))


(defn result-tensor->opencv
  [result-tens]
  (let [[height width n-chan] (m/shape result-tens)
        out-img (opencv/new-mat height width 3 :dtype :uint8)

        host-buffer (cpu/ptr->device-buffer out-img)
        device-buffer (ct/tensor->buffer result-tens)]
    (drv/copy-device->host ct/*stream*
                           device-buffer 0
                           host-buffer 0
                           (* height width n-chan))
    (drv/sync-with-host ct/*stream*)
    out-img))


(defn downsample-img
  [& {:keys [device-type]
      :or {device-type :cpu}}]
  (first
   (vf/tensor-context
    (registry/get-driver device-type)
    :uint8
    (let [mat (opencv/load "test/data/jen.jpg")
          img-tensor (tvm-tm/typed-pointer->tensor mat)
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
          ds-time (with-out-str
                    (time
                     (do
                       (dotimes [iter 10]
                         (resize/area-reduction! img-tensor result
                                                             downsample-fn)
                         (drv/sync-with-host ct/*stream*)))))
          opencv-res (result-tensor->opencv result)
          reference (opencv/new-mat new-height new-width 3 :dtype :uint8)
          ref-time (with-out-str
                     (time
                      (dotimes [iter 10]
                        (opencv/resize-imgproc mat reference :linear))))
          filter-fn (resize/schedule-bilinear-filter-fn
                     :device-type device-type
                     :img-dtype :uint8)
          _ (resize/bilinear-filter! img-tensor result filter-fn)
          classic-time (with-out-str
                         (time
                          (do
                            (dotimes [iter 10]
                              (resize/bilinear-filter! img-tensor result filter-fn)
                              (drv/sync-with-host ct/*stream*)))))
          class-res (result-tensor->opencv result)
          opencv-area (opencv/new-mat new-height new-width 3 :dtype :uint8)
          area-time (with-out-str
                     (time
                      (dotimes [iter 10]
                        (opencv/resize-imgproc mat opencv-area :area))))]
      (opencv/save opencv-res "tvm_area.jpg")
      (opencv/save class-res "tvm_bilinear.jpg")
      (opencv/save reference "opencv_bilinear.jpg")
      (opencv/save opencv-area "opencv_area.jpg")
      {:tvm-area-time ds-time
       :opencv-bilinear-time ref-time
       :tvm-bilinear-time classic-time
       :opencv-area-time area-time}))))


(deftest resize-test
  (doseq [dev-type [:cpu :opencl :cuda]]
    (try
      (println (format "%s:\n%s"
                       dev-type
                       (with-out-str
                         (pp/pprint (downsample-img :device-type dev-type)))))
      (catch Throwable e nil))))
