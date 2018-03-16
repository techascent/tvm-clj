(ns tvm-clj.compute.compile-fn-test
  (:require [tvm-clj.compute.functional-tensor :as ct]
            [tvm-clj.compute.tensor.cpu-functional-tensor]
            ;;unsigned datatype support
            [tvm-clj.compute.host-buffer :as hbuf]
            [clojure.test :refer :all]
            [think.resource.core :as resource]
            [tech.datatype.base :as dtype]
            [tech.datatype.marshal :as marshal]
            [clojure.core.matrix.protocols :as mp]
            [tech.compute.tensor :as compute-tensor]
            [tech.javacpp-datatype :as jcpp-dtype]
            [clojure.core.matrix.macros :refer [c-for]])
  (:import [org.bytedeco.javacpp opencv_core
            opencv_imgcodecs opencv_core$Mat]
           [tech.datatype ByteArrayView]
           ))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn convert-bgr-bytes-to-floats
  "Input is unsigned bytes, so values 0->255, bgr [height width 3].
Output: {:datatype :float32 :shape [3 height width]}, values from -0.5->0.5"
  [input-tensor]
  ;;Move to bgr instead of rgb  Also drop alpha if exists in first place.
  (-> (ct/select input-tensor :all :all [2 1 0])
      ;;transpose to channels first.
      (ct/transpose [2 0 1])
      ;;First actual operation that is compiled.
      ;;Rest of the work.  These should all get rolled into assignment step above.
      (ct/static-cast :float32)
      (ct/div 255)
      (ct/sub 0.5)))


(defn convert-bgr-bytes-to-floats-non-functional
  "In this case, copying isn't the biggest deal"
  [input-tensor]
  (let [input-tensor (-> (compute-tensor/select input-tensor :all :all (compute-tensor/->tensor [2 1 0] :datatype :int32))
                         (compute-tensor/transpose [2 0 1]))
        retval (compute-tensor/new-tensor (compute-tensor/shape input-tensor) :datatype :float32 :init-value nil)]
    (-> (compute-tensor/assign! retval input-tensor)
        (compute-tensor/binary-op! 1.0 retval 1.0 255.0 :/)
        (compute-tensor/binary-op! 1.0 retval 1.0 0.5 :-))))


(defn convert-bgr-bytes-to-floats-by-hand
  [input-tensor]
  (let [^bytes ary-data (.data ^ByteArrayView (:buffer input-tensor))
        [height width n-channels] (ct/shape input-tensor)
        height (long height)
        width (long width)
        n-channels (long n-channels)
        n-elems (long (* height width 3))
        retval (float-array (* 3 height width))]
    (c-for [idx 0 (< idx n-elems) (inc idx)]
           (let [pixel (quot idx n-channels)
                 channel (rem idx n-channels)
                 x-pos (rem pixel width)
                 y-pos (quot pixel width)
                 dest-channel-stride (* x-pos y-pos)]
             (when (< channel 3)
               (aset retval (+ pixel
                               (* (- 2 channel) dest-channel-stride))
                     (float (- (/ (aget ary-data idx)
                                  255.0)
                               0.5))))))
    (compute-tensor/->Tensor (:device input-tensor)
                             {:shape [3 height width]
                              :strides [(* height width) width 1]}
                             (dtype/->view retval))))


(defn convert-floats-to-bgr-bytes
  "Converts to bgr image"
  [img-tensor]
  (-> (ct/add 0.5)
      (ct/mul 255.0)
      (ct/clamp 0 255)
      ;;rgb -> bgr
      (ct/select [2 1 0] :all :all)
      ;;Move channels to last from planar channels first
      (ct/transpose [1 2 0])
      ;;Last operation, uses sophisticated indexing + transposition
      (ct/static-cast :uint8)))


(defn tensor-take
  [n tensor]
  (-> (compute-tensor/as-vector tensor)
      (compute-tensor/select (range n))
      (compute-tensor/to-double-array)
      vec))


(extend-type opencv_core$Mat
  resource/PResource
  (release-resource [item] (.release item))
  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] [(.rows m) (.cols m) (.channels m)])
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))
  mp/PElementCount
  (element-count [m] (apply * (mp/get-shape m)))

  dtype/PDatatype
  ;;For now; I know opencv has more datatypes but whatevs
  (get-datatype [m] :uint8)

  marshal/PContainerType
  (container-type [item] :tvm-host-buffer)

  ;;This allows bulk read/write into the object
  hbuf/PToPtr
  (->ptr [item] (jcpp-dtype/set-pointer-limit-and-capacity
                 (.ptr item)
                 (mp/element-count item)))

  dtype/PCopyRawData
  (copy-raw->item! [item dest dest-offset]
    (marshal/copy! item 0 dest dest-offset (mp/element-count item))
    [dest (+ (long dest-offset) (long (mp/element-count item)))]))



(defn load-image
  [^String filepath]
  (resource/track (opencv_imgcodecs/imread filepath)))


(defn opencv-image-test
  []
  (resource/with-resource-context
    (let [mat (load-image "test/data/jen.jpg")
          img-tensor (compute-tensor/->tensor mat :datatype :int8)]
      img-tensor)))
