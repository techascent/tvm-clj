(ns tech.compute.tvm.compile-test
  (:require [clojure.test :refer :all]
            [tech.resource :as resource]
            [tech.datatype :as dtype]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.dimensions :as ct-dims]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.compute.tvm :as tvm]
            [tvm-clj.api :as api]
            [tech.parallel :as parallel]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [tech.compute.tvm.tensor-math :as tvm-tm]
            [tech.opencv :as opencv]
            [tech.datatype.java-primitive :as primitive]
            [tech.compute :as compute]
            [tech.compute.tensor.defaults :as ct-defaults])
  (:import [java.nio ByteBuffer FloatBuffer]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn convert-bgr-bytes-to-floats-by-hand
  [input-tensor result-tensor]
  (let [^ByteBuffer ary-data (-> (ct/tensor->buffer input-tensor)
                                 primitive/->buffer-backing-store)
        [height width n-channels] (ct/shape input-tensor)
        height (long height)
        width (long width)
        n-channels (long n-channels)
        n-elems (long (* height width 3))
        ^FloatBuffer retval (-> (ct/tensor->buffer result-tensor)
                                primitive/->buffer-backing-store)]
    (parallel/parallel-for idx n-elems
                           (let [pixel (quot idx n-channels)
                                 channel (long (rem idx n-channels))
                                 x-pos (rem pixel width)
                                 y-pos (quot pixel width)
                                 dest-channel-stride (* x-pos y-pos)]
                             (when (< channel 3)
                               (.put retval (+ pixel
                                               (* (- 2 channel) dest-channel-stride))
                                     ;;Make up for lack of unsigned byte support with
                                     ;;bit-and, casting
                                     (float (- (/ (-> (.get ary-data idx)
                                                      int
                                                      (bit-and 0xFF))
                                                  255.0)
                                               0.5))))))
    result-tensor))


(defn bgr-bytes-custom-tvm
  [driver]
  (let [img-width (api/variable "image-width")
        img-height (api/variable "image-height")
        n-channels (api/variable "n-channels")
        input-tensor (api/placeholder [img-height img-width n-channels]
                                      "input-tensor" :dtype :uint8)
        operation (api/compute
                   [(api/min n-channels (int 3)) img-height img-width]
                   (api/tvm-fn
                    [chan y x]
                    (-> (api/tget input-tensor [y x (api/sub 2 chan)])
                        (api/cast :float32)
                        (api/div (float 255.0))
                        (api/sub (float 0.5))))
                   "bgr-types-op")

        schedule (api/create-schedule [operation])
        [chan-axis y-axis x-axis] (:axis operation)
        op-stage (api/->stage schedule operation)
        device-type (tvm/device-type driver)
        [y-outer x-outer y-inner x-inner] (api/stage-tile op-stage y-axis x-axis 8 8)
        arglist [input-tensor (first (api/output-tensors operation))]]
    (api/stage-vectorize op-stage x-inner)
    (if (= :cpu device-type)
      (api/stage-parallel op-stage y-outer)
      (api/stage-bind-gpu op-stage
                          [(api/stage-fuse op-stage [chan-axis y-outer x-outer])]
                          [(api/stage-fuse op-stage [y-inner x-inner])]))
    ;;There are a lot of assumptions in this step.  We aren't providing a bind-map which
    ;;means we expect input to be dense and simple input dimensions
    (println (api/schedule->str schedule arglist :bgr-convert))
    (tvm/schedule->fn driver {:schedule schedule
                              :arglist arglist
                              :name :bgr-convert})))



(defn tensor-take
  [n tensor]
  (-> (ct/as-vector tensor)
      (ct/select (range n))
      (ct/to-double-array)
      vec))


(defn java-by-hand-image-test
  []
  (ct/enable-cpu-tensors!)
  (resource/with-resource-context
    (let [mat (opencv/load "test/data/jen.jpg")
          img-tensor (ct/->tensor mat :datatype :uint8)
          result-tensor (ct/new-tensor (concat [3]
                                               (take 2 (mp/get-shape mat)))
                                       :datatype :float32
                                       :init-value nil)
          time-str (with-out-str
                     (time (convert-bgr-bytes-to-floats-by-hand img-tensor
                                                                result-tensor)))]

      {:result (tensor-take 10 result-tensor)
       :correct (->> (tensor-take 30 img-tensor)
                     (partition 3)
                     (map last)
                     (map #(/ (double (bit-and (int %) 0xFF)) 255.0))
                     (map #(- (double %) 0.5)))
       :time time-str})))


(defn tvm-image-test
  [dev-type]
  (resource/with-resource-context
    (ct-defaults/tensor-driver-context
     (tvm/driver dev-type)
     :float32
      (let [mat (opencv/load "test/data/jen.jpg")
            ;;It would also be possible to do a zero-copy conversion using the
            ;; opencl matrix ptr.
            ;;Note that tvm supports unsigned datatypes.
            nocopy-tensor (tvm/as-cpu-tensor mat)
            ;;If as-tensor fails then we go through the whole upload process.
            img-tensor (if (= dev-type :cpu)
                         nocopy-tensor
                         ;;Example fast path.  Devices can specify if they can copy
                         ;;to another device via device->device copy.  CPU tensors
                         ;;in TVM can be the source or destination of simple copy
                         ;;operations (a copy that boils down to a memcpy).  Note that
                         ;;certain select operations result in dense tensors while
                         ;;others do not.
                         (ct/clone nocopy-tensor))
            result-tensor (ct/new-tensor
                           (concat [3]
                                   (take 2 (mp/get-shape mat)))
                           :datatype :float32
                           :init-value nil)
            tvm-convert-fn (bgr-bytes-custom-tvm (tvm/driver dev-type))
            _ (tvm-convert-fn img-tensor result-tensor)
            time-str (with-out-str
                       (time
                        (do
                          (tvm-convert-fn img-tensor result-tensor)
                          (compute/sync-with-host (ct-defaults/infer-stream {})))))]
        {:result (vec (tensor-take 10 result-tensor))
         :correct (->> (tensor-take 30 img-tensor)
                       (partition 3)
                       (map last)
                       (map #(/ (double %) 255.0))
                       (mapv #(- (double %) 0.5)))
         :time time-str}))))


(defn- is-correct
  [{:keys [result correct] :as retval}]
  (is (m/equals result correct 0.001))
  retval)


(defn- run-test
  [test-fn]
  (-> (test-fn)
      is-correct
      :time))


(deftest time-tests
  []
  (println "hand-coded java took: " (run-test #(java-by-hand-image-test)))

  (println "Compiled (cpu) tensor took:" (run-test #(tvm-image-test :cpu)))

  (println "Compiled (opencl) tensor took:" (run-test #(tvm-image-test :opencl)))

  ;;Not everyone has an nvidia gpu...
  (try
    (println "Compiled (cuda) tensor took:" (run-test #(tvm-image-test :cuda)))
    (catch Throwable e
      nil)))
