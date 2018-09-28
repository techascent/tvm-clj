(ns tvm-clj.image.bilinear-reduce-test
  (:require [clojure.test :refer :all]
            [tvm-clj.image.bilinear-reduce :as bilinear]
            [tech.compute.verify.tensor :as vf]
            [tvm-clj.compute.registry :as registry]
            [tvm-clj.compute.cpu :as cpu]
            [tvm-clj.compile-test :as compile-test]
            [clojure.core.matrix :as m]
            [tech.compute.tensor :as ct]
            [tvm-clj.compute.host-buffer :as hbuf]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [think.resource.core :as resource])
  (:import [org.bytedeco.javacpp opencv_core
            opencv_imgcodecs opencv_core$Mat
            opencv_imgproc opencv_core$Size]))


(defn result-tensor->opencv
  [result-tens]
  (let [[height width n-chan] (m/shape result-tens)
        out-img (resource/track
                 (opencv_core$Mat. height width opencv_core/CV_8UC3))

        host-buffer (cpu/ptr->host-buffer (hbuf/->ptr out-img) :dtype :uint8)
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
    (let [mat (compile-test/load-image "test/data/jen.jpg")
          img-tensor (ct/->tensor mat :datatype :uint8)
          [height width n-chans] (take-last 3 (m/shape img-tensor))
          new-width 512
          ratio (/ (double new-width) width)
          new-height (long (Math/round (* (double height) ratio)))
          result (ct/new-tensor [new-height new-width n-chans] :datatype :uint8)
          downsample-fn (bilinear/schedule-bilinear-reduce-fn
                         :device-type device-type
                         :img-dtype :uint8)
          ;; Call once to take out compilation time
          _ (bilinear/bilinear-reduce! img-tensor result downsample-fn)
          ds-time (with-out-str
                    (time
                     (do
                       (dotimes [iter 10]
                         (bilinear/bilinear-reduce! img-tensor result downsample-fn)
                         (drv/sync-with-host ct/*stream*)))))
          opencv-res (result-tensor->opencv result)
          reference (resource/track (opencv_core$Mat. new-height new-width
                                                      opencv_core/CV_8UC3))
          ref-time (with-out-str
                     (time
                      (dotimes [iter 10]
                        (opencv_imgproc/resize mat reference (opencv_core$Size.
                                                              new-width
                                                              new-height)
                                               0.0 0.0 (opencv_imgproc/CV_INTER_LINEAR)))))
          filter-fn (bilinear/schedule-bilinear-filter-fn
                     :device-type device-type
                     :img-dtype :uint8)
          _ (bilinear/bilinear-filter! img-tensor result filter-fn)
          classic-time (with-out-str
                         (time
                          (do
                            (dotimes [iter 10]
                              (bilinear/bilinear-filter! img-tensor result filter-fn)
                              (drv/sync-with-host ct/*stream*)))))
          class-res (result-tensor->opencv result)]
      (opencv_imgcodecs/imwrite "tvm_correct.jpg" opencv-res)
      (opencv_imgcodecs/imwrite "tvm_classic.jpg" class-res)
      (opencv_imgcodecs/imwrite "opencv_classic.jpg" reference)
      {:tvm-correct-time ds-time
       :opencv-classic-time ref-time
       :tvm-classic-time classic-time}))))
