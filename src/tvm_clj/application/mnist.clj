(ns tvm-clj.application.mnist
  (:require [tech.v3.tensor :as dtt]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.monotonic-range :as dtype-range]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype.mmap :as mmap]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.libs.buffered-image :as bufimg]
            [tvm-clj.application.kmeans :as kmeans]
            [clojure.tools.logging :as log]))

(def train-fnames {:data "train-images-idx3-ubyte"
                   :labels "train-labels-idx1-ubyte"})
(def test-fnames {:data "t10k-images-idx3-ubyte"
                  :labels "t10k-labels-idx1-ubyte"})


(def ^{:tag 'long} img-width 28)
(def ^{:tag 'long} img-height 28)
(def ^{:tag 'long} img-size (* img-width img-height))


(defn save-mnist-tensor-as-img
  ([tensor fname]
   (-> (dtt/reshape tensor [img-height img-width])
       (dtype/copy! (bufimg/new-image img-height img-width :byte-gray))
       (bufimg/save! fname))))


(defn mmap-file
  [fname]
  (-> (mmap/mmap-file (format "test/data/%s" fname))
      (native-buffer/set-native-datatype :uint8)))


(defn load-data
  [fname]
  (let [fdata (mmap-file fname)
        n-images (long (quot (dtype/ecount fdata) img-size))
        leftover (rem (dtype/ecount fdata)
                      img-size)]
    (-> (dtype/sub-buffer fdata leftover)
        (dtt/reshape [n-images img-height img-width]))))


(defn load-labels
  [fname]
  (-> (mmap-file fname)
      (dtype/sub-buffer 8)))


(defn load-dataset
  [dataset ds-name]
  ;;Data is an [n-images height width] tensor
  (log/infof "Loading %s dataset" ds-name)
  {:data (load-data (dataset :data))
   :labels (load-labels (dataset :labels))})


;;Datasets are maps of class-label->tensor
(defonce train-ds (load-dataset train-fnames "train"))
(defonce test-ds (load-dataset test-fnames "test"))


(defn reshape-data
  "Reshape images to be a 2d matrix of rows where each image is one row."
  [data]
  (let [[n-images height width] (dtype/shape data)]
    (dtt/reshape data [n-images (* (long height) (long width))])))


(defn train-kmeans-per-label
  [n-per-label & [{:keys [seed n-iters] :as options}]]
  (kmeans/train-per-label (reshape-data (:data train-ds))
                          (:labels train-ds)
                          n-per-label options))


(defn save-centers-as-images!
  [centers]
  (let [n-centers (long (first (dtype/shape centers)))]
    (doseq [idx (range n-centers)]
      (-> (centers idx)
          (save-mnist-tensor-as-img (format "center-%d.png" idx))))))

(defn kmeans->histograms
  "Takes the output of `train-kmeans` and returns a histogram of original labels for each learned center."
  [{:keys [assigned-centers]}]
  (->> (for [[idx assigned-center] (map-indexed vector assigned-centers)]
         (let [idx->label (fn [i]
                            (->> (map-indexed vector (:labels train-ds))
                                 (map (fn [[label-i [minimum maximum]]]
                                        (when (and (<= minimum i) (< i maximum))
                                          label-i)))
                                 (remove nil?)
                                 (first)))]
           {:label (idx->label idx)
            :assigned-center assigned-center}))
       (group-by :assigned-center)
       (map (fn [[_ item-seq]]
              (frequencies (map :label item-seq))))
       (sort-by (fn [hist] (ffirst (sort-by second > hist))))
       (map (fn [center]
              (for [i (range 10)]
                (get center i 0))))
       (dtt/ensure-tensor)))


(defn confusion-matrix
  [labels predictions]
  (let [retval (dtt/new-tensor [10 10] :datatype :int64)]
    (doseq [[label pred] (map vector labels predictions)]
      (.ndAccumPlusLong retval label pred 1)
      (.ndAccumPlusLong retval pred label 1))
    retval))


(defn test-kmeans
  [& [n-centers]]
  (let [n-centers (or n-centers 10)]
    (let [model (kmeans/train-per-label (reshape-data (:data train-ds))
                                        (:labels train-ds)
                                        n-centers)
          prediction-data (time (kmeans/predict-per-label (reshape-data (:data test-ds))
                                                          model))
          labels (:labels test-ds)
          predictions (:label-indexes prediction-data)]
      {:accuracy (/ (dfn/sum (dfn/eq labels predictions))
                    (dtype/ecount predictions))
       :confusion-matrix (confusion-matrix labels predictions)})))



(defn test-n-center-predictors
  [n-centers]
  (->> (range 1 (inc n-centers))
       (mapv (fn [idx]
               (let [model (kmeans/train-per-label (reshape-data (:data train-ds))
                                                   (:labels train-ds)
                                                    idx)
                     prediction-data (kmeans/predict-per-label (reshape-data (:data test-ds))
                                                               model)
                     labels (:labels test-ds)
                     predictions (:label-indexes prediction-data)]
                 {:accuracy (/ (dfn/sum (dfn/eq labels predictions))
                               (dtype/ecount predictions))
                  :confusion-matrix (confusion-matrix labels predictions)})))))
