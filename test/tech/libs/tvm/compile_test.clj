(ns tech.libs.tvm.compile-test
  (:require [clojure.test :refer :all]
            [tech.resource :as resource]
            [tech.v2.datatype :as dtype]
            [tech.v2.datatype.functional :as dfn]
            [tech.v2.datatype.typecast :as typecast]
            [tech.v2.tensor :as dtt]
            [tech.v2.tensor.impl :as dtt-impl]
            [tech.v2.tensor.dimensions :as dtt-dims]
            [tech.libs.tvm :as tvm]
            [tvm-clj.api :as api]
            [tech.opencv :as opencv]
            [tech.compute :as compute]
            [tech.compute.driver :as drv]
            [tech.libs.tvm.cpu]
            [tech.libs.tvm.gpu])
  (:import [tech.v2.datatype FloatReader]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


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
    ;;There are a lot of assumptions in this step.  We aren't providing a bind-map
    ;;which means we expect input to be dense and simple input dimensions
    (println (api/schedule->str schedule arglist :bgr-convert))
    (tvm/schedule->fn driver {:schedule schedule
                              :arglist arglist
                              :name :bgr-convert})))


(defn tensor-take
  [n tensor]
  (->> (dtype/->reader tensor)
       (take n)
       (dtype/->double-array)
       vec))


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


(defn clone-to-device
  [input-tens dev-type & [stream]]
  (let [tvm-device (tvm/device dev-type)
        stream (or stream (compute/default-stream tvm-device))
        dev-buf (compute/allocate-device-buffer tvm-device
                                                (dtype/ecount input-tens)
                                                (dtype/get-datatype input-tens))
        input-tens (if (and (dtt/dimensions-dense? input-tens)
                            (drv/acceptable-host-buffer?
                             (drv/get-driver tvm-device)
                             input-tens))
                     input-tens
                     (dtt/clone input-tens :container-type :native-buffer))]
    (compute/copy-host->device input-tens 0 dev-buf 0 (dtype/ecount input-tens)
                               :stream stream)
    (dtt-impl/construct-tensor dev-buf
                               (dtt-dims/dimensions (dtype/shape input-tens)))))


(defn clone-to-host
  [device-tens dev-type & [stream]]
  (let [tvm-device (tvm/device dev-type)
        stream (or stream (compute/default-stream tvm-device))
        dev-buf (compute/allocate-host-buffer (drv/get-driver tvm-device)
                                              (dtype/ecount device-tens)
                                              (dtype/get-datatype device-tens))]
    (compute/copy-device->host device-tens 0 dev-buf 0 (dtype/ecount device-tens)
                               :stream stream)
    (drv/sync-with-host stream)
    (dtt-impl/construct-tensor dev-buf
                               (dtt-dims/dimensions (dtype/shape device-tens)))))


(defn tvm-image-test
  [dev-type]
  (let [mat (opencv/load "test/data/jen.jpg")
        ;;It would also be possible to do a zero-copy conversion using the
        ;; opencl matrix ptr.
        ;;Note that tvm supports unsigned datatypes.
        nocopy-tensor (dtt/ensure-tensor mat)
        ;;If as-tensor fails then we go through the whole upload process.
        img-tensor (if (= dev-type :cpu)
                     nocopy-tensor
                     ;;Example fast path.  Devices can specify if they can copy
                     ;;to another device via device->device copy.  CPU tensors
                     ;;in TVM can be the source or destination of simple copy
                     ;;operations (a copy that boils down to a memcpy).  Note that
                     ;;certain select operations result in dense tensors while
                     ;;others do not.
                     (clone-to-device nocopy-tensor dev-type))
        result-shape (vec (concat [3]
                                  (take 2 (dtype/shape mat))))
        device (tvm/device dev-type)
        result-ecount (apply * result-shape)
        result-tensor
        (dtt-impl/construct-tensor
         (drv/allocate-device-buffer device
                                     result-ecount
                                     :float32
                                     {})
         (dtt-dims/dimensions result-shape))
        stream (compute/default-stream device)
        tvm-convert-fn (bgr-bytes-custom-tvm (tvm/driver dev-type))
        _ (tvm-convert-fn img-tensor result-tensor)
        time-str (with-out-str
                   (time
                    (do
                      (tvm-convert-fn img-tensor result-tensor)
                      (compute/sync-with-host stream))))]
    {:result (vec (tensor-take 10 (clone-to-host result-tensor dev-type)))
     :correct (->> (tensor-take 30 img-tensor)
                   (partition 3)
                   (map last)
                   (map #(/ (double %) 255.0))
                   (mapv #(- (double %) 0.5)))
     :time time-str}))


(defn- is-correct
  [{:keys [result correct] :as retval}]
  (is (dfn/equals result correct 0.001))
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
