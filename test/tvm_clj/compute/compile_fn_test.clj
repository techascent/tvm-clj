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
            [clojure.core.matrix.macros :refer [c-for]]
            [tvm-clj.compute.compile-fn :as compiler]
            [tvm-clj.compute.base :as comp-base]
            [tech.compute.driver :as drv]
            [think.parallel.core :as parallel])
  (:import [org.bytedeco.javacpp opencv_core
            opencv_imgcodecs opencv_core$Mat]
           [tech.datatype ByteArrayView FloatArrayView]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn convert-bgr-bytes-to-floats
  "Input is unsigned bytes, so values 0->255, bgr [height width 3].
Output: {:datatype :float32 :shape [3 height width]}, values from -0.5->0.5"
  [input-tensor]
  ;;Move to bgr instead of rgb  Also drop alpha if exists in first place.
  (-> (ct/select input-tensor :all :all [2 1 0])
      ;;Image is now planar; so a plane of b, a plane of g, and a plane of r.
      (ct/transpose [2 0 1])
      ;;First actual operation that is compiled.
      (ct/static-cast :float32)
      ;;Rest of the work.  These should all get rolled into assignment step above.
      (ct/div 255.0)
      (ct/sub 0.5)))


(defn compile-bgr-bytes
  [& {:keys [driver-name]
      :or {driver-name :cpu}}]
  (let [graph (-> (compiler/compute-graph)
                  (compiler/make-variable :image-width)
                  (compiler/make-variable :image-height)
                  (compiler/make-variable :image-channels)
                  (compiler/make-tensor-and-buffer :input [:image-height :image-width :image-channels] :dtype :uint8))
        input-tensor (compiler/get-tensor graph :input)]
    (assoc (compiler/compile-fn (comp-base/get-driver driver-name) graph convert-bgr-bytes-to-floats input-tensor)
           :driver driver-name)))


(defn convert-bgr-bytes-to-floats-by-hand
  [input-tensor result-tensor]
  (let [^bytes ary-data (.data ^ByteArrayView (:buffer input-tensor))
        [height width n-channels] (ct/shape input-tensor)
        height (long height)
        width (long width)
        n-channels (long n-channels)
        n-elems (long (* height width 3))
        ^floats retval (.data ^FloatArrayView (:buffer result-tensor))]
    (parallel/parallel-for idx n-elems
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
    result-tensor))



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


(defn java-tensor-ops-image-test
  []
  (resource/with-resource-context
    (let [mat (load-image "test/data/jen.jpg")
          img-tensor (compute-tensor/->tensor mat :datatype :int8)
          result-tensor (compute-tensor/new-tensor (concat [3]
                                                           (take 2 (mp/get-shape mat)))
                                                   :datatype :float32
                                                   :init-value nil)]
      (convert-bgr-bytes-to-floats img-tensor)
      (time (convert-bgr-bytes-to-floats img-tensor)))))


(defn java-by-hand-image-test
  []
  (resource/with-resource-context
    (let [mat (load-image "test/data/jen.jpg")
          img-tensor (compute-tensor/->tensor mat :datatype :int8)
          result-tensor (compute-tensor/new-tensor (concat [3]
                                                           (take 2 (mp/get-shape mat)))
                                                   :datatype :float32
                                                   :init-value nil)]
      (convert-bgr-bytes-to-floats-by-hand img-tensor result-tensor)
      (time (convert-bgr-bytes-to-floats-by-hand img-tensor result-tensor)))))


(defn tvm-image-test
  []
  (resource/with-resource-context
    (compute-tensor/with-stream (drv/default-stream
                                 (comp-base/get-device :cpu 0))
      (let [mat (load-image "test/data/jen.jpg")
            ;;It would also be possible to do a zero-copy conversion using the
            ;; opencl matrix ptr.
            ;;Note that tvm supports unsigned datatypes.
            img-tensor (compute-tensor/->tensor mat :datatype :uint8)
            result-tensor (compute-tensor/new-tensor (concat [3]
                                                             (take 2 (mp/get-shape mat)))
                                                     :datatype :float32
                                                     :init-value nil)
            ;; result-tensor (compute-tensor/new-tensor (mp/get-shape mat)
            ;;                                          :datatype :float32
            ;;                                          :init-value nil)
            {:keys [inputs outputs fn!]} (compile-bgr-bytes)
            ;;This is abit careless but I know the results of the compilation process
            arg-map {(get-in inputs [0 :id]) img-tensor
                     (get-in outputs [0 :id]) result-tensor}]
        (time (fn! arg-map))
        {:result (tensor-take 10 result-tensor)
         :correct (->> (tensor-take 30 img-tensor)
                       (partition 3)
                       (map last)
                       (map #(/ (double %) 255.0))
                       (map #(- (double %) 0.5)))}))))


(defn time-tests
  []
  (println "java tensor ops took: " (with-out-str
                                      (java-tensor-ops-image-test)))

  (println "hand-coded java took: " (with-out-str
                                      (java-by-hand-image-test)))

  (println "Compiled tensor took:" (with-out-str
                                     (tvm-image-test))))