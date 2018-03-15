(ns tvm-clj.compute.compile-fn-test)

(def img-width 512)
(def img-height 512)


(defn convert-bgr-bytes-to-floats
  "Input is unsigned bytes, so values 0->255, bgr [height width 3].
Output: {:datatype :float32 :shape [3 height width]}, values from -0.5->0.5"
  [input-tensor]
  (let [result (ct/new-tensor  (concat [3] (take 2 (ct/shape input-tensor)))
                               :datatype :float32)
        reshape-indexes (ct/->tensor [2 1 0]
                                     :datatype :int32
                                     :allocation :const)
        input-reshaped (-> (ct/select :all :all reshape-indexes)
                           (ct/transpose [2 0 1]))]
    (-> (ct/=! result input-reshaped)
        (ct//! 255)
        (ct/+! -0.5))))

(defn convert-floats-to-bgr-bytes
  "Converts to bgr image"
  [img-tensor]
  (let [result (ct/new-tensor (concat (drop 1 (ct/shape input-tensor))
                                      [(last input-tensor)]))
        reshape-indexes (ct/->tensor [2 1 0]
                                     :datatype :int32
                                     :allocation :const)
        img-tensor (ct/select reshape-indexes reshape-indexes :all :all)]
    (-> (ct/+! ))
    )
  )

(def cpu-compiled-image->floats (tvm/compile-cpu! convert-image-to-floats
                                                  (ct/bind-tensor [img-height img-width 4])))

(let [test-image (-> (opencv/imload (io/resource "test-image.png"))
                     ;;Zero memcopy; uses opencv mat pointer
                     (tvm-cpu/as-tensor))
      [height height n-channels] (ct/shape test-image)
      workspace (ct/workspace)
      reference (ct/to-double-array (ct/with-workspace workspace
                                      (convert-image-to-floats test-image)))]
  (assert! (and (= height img-height)
                (= width img-width)))
  (resource/with-resource-context
    (time (dotimes [iter 1000]
            (ct/with-workspace workspace
              (convert-image-to-floats test-image)))))
  (resource/with-resource-context
    (time (dotimes [iter 1000]
            (ct/with-workspace workspace
              (cpu-compiled-image->floats test-image))))))
