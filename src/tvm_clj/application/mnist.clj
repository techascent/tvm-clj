(ns tvm-clj.applications.mnist
  (:require [tech.v3.tensor :as dtt]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype.mmap :as mmap]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.libs.buffered-image :as bufimg]))

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
        n-images (long (/ (dtype/ecount fdata) img-size))]
    (-> (dtt/reshape fdata [n-images img-height img-width])
        (dtt/rotate [0 0 13]))))


(defn load-labels
  [fname]
  (-> (mmap-file fname)
      (dtype/sub-buffer 8)))


(defn load-dataset
  [dataset]
  (let [data (load-data (dataset :data))
        labels (load-labels (dataset :labels))]
    (->> (argops/arggroup labels)
         (map (fn [[label idx-list]]
                [label (-> (dtt/select data idx-list)
                           (dtt/clone :container-type :native-heap))]))
         (sort-by first)
         (into {}))))


(defonce train-ds (load-dataset train-fnames))
(defonce test-ds (load-dataset train-fnames))
