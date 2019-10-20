(ns tech.libs.tvm.compile-test
  (:require [clojure.test :refer :all]
            [tech.resource :as resource]
            [tech.v2.datatype :as dtype]
            [tech.v2.datatype.functional :as dfn]
            [tech.v2.datatype.typecast :as typecast]
            [tech.v2.tensor :as dtt]
            [tech.libs.tvm :as tvm]
            [tvm-clj.api :as api]
            [tech.opencv :as opencv]
            [tech.compute :as compute]
            [tech.compute.context :as compute-ctx]
            [tech.compute.tensor :as compute-tens]
            [tech.libs.tvm.cpu]
            [tech.libs.tvm.gpu]
            [clojure.tools.logging :as log])
  (:import [tech.v2.datatype FloatReader]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn tensor-take
  [n tensor]
  (->> (dtype/->reader tensor)
       (take n)
       (dtype/->double-array)
       vec))


(defn convert-bgr-bytes-to-floats-by-hand
  [input-tensor result-tensor]
  (let [ary-data (typecast/datatype->reader :float32 input-tensor)
        [_height _width n-channels] (dtype/shape input-tensor)
        n-channels (long n-channels)
        n-elems (dtype/ecount result-tensor)
        result-reader
        (reify FloatReader
          (lsize [rdr] n-elems)
          (read [rdr idx]
            (let [chan (rem idx 3)
                  pix (quot idx 3)]
              (-> (.read ary-data (+ (* pix n-channels)
                                     (- n-channels chan 1)))
                  (/ 255.0)
                  (- 0.5)
                  float))))]
    (dtype/copy! result-reader result-tensor)))


(defn java-by-hand-image-test
  []
  (resource/stack-resource-context
    (let [mat (opencv/load "test/data/jen.jpg")
          result-tensor (dtt/new-tensor (vec (concat [3]
                                                     (take 2 (dtype/shape mat))))
                                       :datatype :float32
                                       :init-value nil)
          time-str (with-out-str
                     (time (convert-bgr-bytes-to-floats-by-hand mat result-tensor)))]

      {:result (tensor-take 10 result-tensor)
       :correct (->> (tensor-take 30 mat)
                     (partition 3)
                     (map last)
                     (map #(/ (double (bit-and (int %) 0xFF)) 255.0))
                     (map #(- (double %) 0.5)))
       :time time-str})))


(defn bgr-bytes-custom-tvm
  [& [driver]]
  (let [driver (or driver (compute-ctx/default-driver))
        img-width (api/variable "image-width")
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
    ;;There are a lot of assumptions in this step.  We aren't providing a bind-map
    ;;which means we expect input to be dense and simple input dimensions
    (println (api/schedule->str schedule arglist :bgr-convert))
    (tvm/schedule->fn driver {:schedule schedule
                              :arglist arglist
                              :name :bgr-convert})))


(defn tvm-image-test
  [dev-type]
  (resource/stack-resource-context
   (compute-ctx/with-context
     {:driver (tvm/driver dev-type)}
     (try
       (let [mat (opencv/load "test/data/jen.jpg")
             ;;It would also be possible to do a zero-copy conversion using the
             ;; opencl matrix ptr.
             ;;If the device is a cpu device, then there is no upload step.
             input-tensor (compute-tens/ensure-device mat)
             result-shape (vec (concat [3]
                                       (take 2 (dtype/shape mat))))
             result-tensor (compute-tens/new-tensor result-shape
                                                    {:datatype :float32})
             tvm-convert-fn (bgr-bytes-custom-tvm)
             _ (tvm-convert-fn input-tensor result-tensor)
             _ (compute/sync-with-host)
             time-str (with-out-str
                        (time
                         (do
                           (tvm-convert-fn input-tensor result-tensor)
                           (compute/sync-with-host))))]
         {:result (vec (tensor-take 10 (compute-tens/ensure-host result-tensor)))
          :correct (->> (tensor-take 30 (compute-tens/ensure-host input-tensor))
                        (partition 3)
                        (map last)
                        (map #(/ (double %) 255.0))
                        (mapv #(- (double %) 0.5)))
          :time time-str})
       (catch Throwable e
         (log/error e))))))


(defn- is-correct
  [{:keys [result correct] :as retval}]
  (when retval
    (is (dfn/equals result correct 0.001)))
  retval)


(defn- run-test
  [test-fn]
  (-> (test-fn)
      is-correct
      :time))


(deftest time-tests
  []
  (println "hand-coded java took: " (:time (java-by-hand-image-test)))

  (println "Compiled (cpu) tensor took:" (run-test #(tvm-image-test :cpu)))

  (println "Compiled (opencl) tensor took:" (run-test #(tvm-image-test :opencl)))

  ;;Not everyone has an nvidia gpu...
  (try
    (println "Compiled (cuda) tensor took:" (run-test #(tvm-image-test :cuda)))
    (catch Throwable e
      nil)))
