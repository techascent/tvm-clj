(ns tvm-clj.image.bilinear-reduce-test
  (:require [clojure.test :refer :all]
            [tvm-clj.image.bilinear-reduce :as bilinear]
            [tech.compute.verify.tensor :as vf]
            [tvm-clj.compute.registry :as registry]
            [tvm-clj.compute.cpu :as cpu]
            [tvm-clj.compile-test :as compile-test]
            [clojure.core.matrix :as m]
            [tech.compute.tensor :as ct]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [think.resource.core :as resource]
            [tech.typed-pointer :as typed-pointer]
            [tech.opencv :as opencv]
            [tvm-clj.compute.tensor-math :as tvm-tm]))


(defn result-tensor->opencv
  [result-tens]
  (let [[height width n-chan] (m/shape result-tens)
        out-img (opencv/new-mat height width 3 :dtype :uint8)

        host-buffer (cpu/ptr->device-buffer (typed-pointer/->ptr out-img) :dtype :uint8)
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
          downsample-fn (bilinear/schedule-correct-reduction
                         :device-type device-type
                         :img-dtype :uint8)
          ;; Call once to take out compilation time
          _ (bilinear/correct-linear-reduction! img-tensor result downsample-fn)
          ds-time (with-out-str
                    (time
                     (do
                       (dotimes [iter 10]
                         (bilinear/correct-linear-reduction! img-tensor result
                                                             downsample-fn)
                         (drv/sync-with-host ct/*stream*)))))
          opencv-res (result-tensor->opencv result)
          reference (opencv/new-mat new-height new-width 3 :dtype :uint8)
          ref-time (with-out-str
                     (time
                      (dotimes [iter 10]
                        (opencv/resize-imgproc mat reference :linear))))
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
      (opencv/save opencv-res "tvm_correct.jpg")
      (opencv/save class-res "tvm_classic.jpg")
      (opencv/save reference "opencv_classic.jpg")
      {:tvm-correct-time ds-time
       :opencv-classic-time ref-time
       :tvm-classic-time classic-time}))))
