(ns tvm-clj.applications.mnist
  (:require [tech.v3.tensor :as dtt]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.monotonic-range :as dtype-range]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype.mmap :as mmap]
            [tech.v3.datatype.argops :as argops]
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


(defn save-tensor-as-img
  ([tensor fname]
   (let [[img-height img-width] (dtype/shape tensor)]
     (-> (dtype/copy! tensor (bufimg/new-image img-height img-width :byte-gray))
         (bufimg/save! fname)))))


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
  (let [data (load-data (dataset :data))
        labels (load-labels (dataset :labels))
        sorted-label-indexes (argops/argsort labels)
        ;;Order data by label so that we can just use range
        ;;offsets
        data (-> (dtt/select data sorted-label-indexes)
                 ;;One block of data all contiguous
                 (dtt/clone :container-type :native-heap))
        labels (dtype/indexed-buffer sorted-label-indexes labels)
        result
        {:data data
         :labels (->> (argops/arggroup labels)
                      ;;We know that idx-list is ordered (arggroup guarantees this)and that it is
                      ;;contiguous by our sort above.
                      (map (fn [[label idx-list]]
                             ;;make the labels print nicely
                             [label [(first idx-list)
                                     (inc (last idx-list))]]))
                      (sort-by first)
                      (mapv second))}]
    (log/infof "Finished loading %s dataset" ds-name)
    result))


;;Datasets are maps of class-label->tensor
(defonce train-ds (load-dataset train-fnames "train"))
(defonce test-ds (load-dataset test-fnames "test"))


(defn train-kmeans
  [n-centers & [{:keys [seed n-iters] :as options}]]
  (errors/when-not-errorf
   (>= n-centers 10)
   "Must train at least 10 centers: received %d" n-centers)
  (let [{:keys [data labels]} train-ds
        [n-images height width] (dtype/shape data)
        dataset (dtt/reshape data [n-images (* (long height) (long width))])
        {:keys [centers scores assigned-centers] :as result}
        (kmeans/kmeans++ dataset n-centers (assoc options :n-iters n-iters))]
    result))


(defn train-kmeans-per-label
  [n-per-label & [{:keys [seed n-iters] :as options}]]
  (let [{:keys [data labels]} train-ds
        result-seq (->> labels
                        (mapv (fn [[idx-start past-idx-end]]
                                ;;Tensor selection from contiguous data of a range with an increment of 1
                                ;;is guaranteed to produce contiguous data
                                (let [per-label-data (dtt/select data (range idx-start past-idx-end))
                                      [n-images height width] (dtype/shape per-label-data)
                                      train-data (dtt/reshape per-label-data [n-images (* height width)])]
                                  (kmeans/kmeans++ train-data n-per-label options)))))
        centers (dtype/coalesce! (dtt/new-tensor [(* n-per-label (count labels))
                                                  (* img-height img-width)])
                                 (map :centers result-seq))
        scores (map-indexed vector (:scores result-seq))]
    {:centers centers
     :scores scores}))


(defn save-centers-as-images!
  [centers]
  (let [n-centers (long (first (dtype/shape centers)))]
    (doseq [idx (range n-centers)]
      (-> (centers idx)
          (dtt/reshape [img-height img-width])
          (save-tensor-as-img (format "center-%d.png" idx))))))
