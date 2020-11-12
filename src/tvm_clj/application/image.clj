(ns tvm-clj.application.image
  "Image resize algorithms showing somewhat nontrivial application
  of TVM operators."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.datatype.casting :as casting]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.tensor :as dtt]
            [tech.v3.tensor.pprint :as tens-pp]
            [tech.v3.tensor.dimensions :as dims]
            [tech.v3.tensor.dimensions.analytics :as dims-analytics]
            [tech.v3.libs.buffered-image :as bufimg]
            )
  (:import [tech.v3.datatype NDBuffer ObjectReader]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn- n-img-shape
  "Normalize shape to dimensions."
  [shape-vec]
  (case (count shape-vec)
    2 (vec (concat [1] shape-vec [1]))
    3 (vec (concat [1] shape-vec))
    4 shape-vec))


(defn- to-long-round
  ^long [value]
  (long (Math/round (double value))))


(defn- to-long-ceil
  ^long [value]
  (long (Math/ceil value)))


(defn- area-filter-pixel-size
  "The size of the destination pixel in input pixels on one axis."
  ^long [in-size out-size]
  (let [temp (/ (double in-size)
                (double out-size))]
    (long
     (if (= (Math/floor temp)
            temp)
       temp
       ;;If it is fractional then it could overlap on either side
       ;;at the same time.
       (Math/floor (+ temp 2))))))


(defn area-reduction!
  [input output area-fn]
  (let [[in-height in-width in-chan] (dtype/shape input)
        [out-height out-width out-chan] (dtype/shape output)
        filter-height (/ (double in-height)
                         (double out-height))
        filter-width (/ (double in-width)
                        (double out-width))
        kernel-height (area-filter-pixel-size in-height out-height)
        kernel-width (area-filter-pixel-size in-width out-width)]
    (area-fn input output
                 kernel-width filter-width
                 kernel-height filter-height)
    output))


(defn- clamp
  ^double [^double value ^double val_min ^double val_max]
  (-> (min value val_max)
      (max val_min)))


(defn- clamp-long
  ^long [^long value ^long val_min ^long val_max]
  (-> (min value val_max)
      (max val_min)))


(defn- compute-tensor
  [output-shape per-pixel-op datatype]
  (let [dims (dims/dimensions output-shape)
        output-shape (long-array output-shape)
        n-dims (count output-shape)
        n-elems (long (apply * output-shape))
        shape-chan (aget output-shape (dec n-dims))
        shape-x (when (> n-dims 1)
                  (aget output-shape (dec (dec n-dims))))]
    (dtt/construct-tensor
     (reify ObjectReader
       (elemwiseDatatype [rdr] datatype)
       (lsize [rdr] n-elems)
       (readObject [rdr idx]
         (case n-dims
           1 (per-pixel-op idx)
           2 (per-pixel-op (quot idx shape-chan)
                           (rem idx shape-chan))
           3 (let [c (rem idx shape-chan)
                   xy (quot idx shape-chan)
                   x (rem xy (long shape-x))
                   y (quot xy (long shape-x))]
               (per-pixel-op y x c))
           (let [local-data (long-array n-dims)]
             (loop [fwd-dim-idx 0
                    idx idx]
               (when (and (> idx 0) (< fwd-dim-idx n-dims))
                 (let [dim-idx (- n-dims fwd-dim-idx 1)
                       local-shape (aget output-shape dim-idx)
                       cur (rem idx local-shape)]
                   (aset local-data dim-idx cur)
                   (recur (unchecked-inc fwd-dim-idx)
                          (quot idx (aget output-shape dim-idx))))))
             (apply per-pixel-op local-data)))))
     dims)))


(defn- src-coord
  ^long [^long dest-coord ^long kernel-idx ^long kernel-width ^double out-over-in]
  (- (+ (Math/round (/ dest-coord out-over-in))
        kernel-idx)
     (quot kernel-width 2)))



(defn jvm-area-resize
  [input output-shape]
  (let [[^long in-height ^long in-width n-chan] (dtype/shape input)
        [^long out-height ^long out-width n-chan] output-shape
        input (dtt/ensure-tensor input)
        max-idx-x (dec in-width)
        max-idx-y (dec in-height)
        x-ratio (double (/ (double out-width) in-width))
        y-ratio (double (/ (double out-width) in-height))
        x-kernel-width (/ 1.0 x-ratio)
        y-kernel-width (/ 1.0 y-ratio)
        divisor (* x-ratio y-ratio)
        reducer (fn [^double accum ^double input]
                  (+ accum input))
        identity-value 0.0]
    (println {:divisor divisor
              :x-ratio x-ratio
              :y-ratio y-ratio
              :x-kernel-width x-kernel-width
              :y-kernel-width y-kernel-width})
    (compute-tensor [out-height out-width n-chan]
                    (fn [^long y ^long x ^long c]
                      (* divisor
                         (double
                          (loop [k-idx-y 0
                                 outer-sum identity-value]
                            (if (< k-idx-y y-kernel-width)
                              (recur (unchecked-inc k-idx-y)
                                     (double
                                      (loop [k-idx-x 0
                                             inner-sum outer-sum]
                                        (if (< k-idx-x x-kernel-width)
                                          (let [src-coord-x (clamp-long
                                                             (src-coord x k-idx-x x-kernel-width x-ratio)
                                                             0
                                                             max-idx-x)
                                                src-coord-y (clamp-long
                                                             (src-coord y k-idx-y y-kernel-width y-ratio)
                                                             0
                                                             max-idx-y)]
                                            (recur (unchecked-inc k-idx-x)
                                                   (double
                                                    (reducer inner-sum (.ndReadDouble input src-coord-y
                                                                                      src-coord-x c)))))
                                          inner-sum))))
                              outer-sum)))))
                    :float64)))


(defn area-resize
  [src-img target-width resize-fn]
  (let [input (dtt/ensure-tensor src-img)
        [^long h ^long w ^long c] (dtype/shape input)
        ratio (/ (double target-width) w)
        new-height (to-long-round (* h ratio))
        dst-img (bufimg/new-image new-height target-width
                                  (bufimg/image-type src-img))
        output (dtt/ensure-tensor dst-img)]
    (area-reduction! input output jvm-area-resize!)
    dst-img))


(comment
  (def input-img (bufimg/load "test/data/jen.jpg"))
  (def )
  )
