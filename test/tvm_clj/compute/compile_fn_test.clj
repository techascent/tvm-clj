(ns tvm-clj.compute.compile-fn-test)

(def img-width 512)
(def img-height 512)


(defn convert-bgr-bytes-to-floats
  "Input is unsigned bytes, so values 0->255, bgr [height width 3].
Output: {:datatype :float32 :shape [3 height width]}, values from -0.5->0.5"
  [input-tensor]
  ;;Move to bgr instead of rgb  Also drop alpha if exists in first place.
  (-> (ct/select :all :all [2 1 0])
      ;;transpose to channels first.
      (ct/transpose [2 0 1])
      ;;First actual operation that is compiled.
      (ct/convert :float32)
      ;;Rest of the work.  These should all get rolled into assignment step above.
      (ct// 255)
      (ct/- 0.5)))


(defn convert-floats-to-bgr-bytes
  "Converts to bgr image"
  [img-tensor]
  (-> (ct/+ 0.5)
      (ct/* 255.0)
      (ct/clamp 0 255)
      ;;rgb -> bgr
      (ct/select [2 1 0] :all :all)
      ;;Move channels to last from planar channels first
      (ct/transpose [1 2 0])
      ;;Last operation, uses sophisticated indexing + transposition
      (ct/convert :uint8)))


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
