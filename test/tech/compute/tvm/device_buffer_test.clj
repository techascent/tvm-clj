(ns tvm-clj.compute.device-buffer-test
  (:require [tech.compute.tvm :as tvm]
            [tech.datatype.base :as dtype]
            [tech.compute.driver :as drv]
            [tech.resource :as resource]
            [clojure.test :refer :all]))


(deftest host-read-write
  (testing "Efficient bulk transfer to/from tvm device buffer"
    (with-bindings {#'dtype/*error-on-slow-path* true}
      (resource/with-resource-context
        (let [test-data [1 2 3 253 254 255]
              short-buf (short-array test-data)
              ;;This test has a hidden difficulty.  255 is not representable in
              ;;a java byte.
              test-buf (tvm/make-cpu-device-buffer :uint8 6)
              end-buf (double-array 6)]

          (dtype/copy! short-buf 0 test-buf 0 6)
          (dtype/copy! test-buf 0 end-buf 0 6)
          (is (= test-data
                 (mapv int end-buf))))))))


(deftest host-buffer-offset
  (testing "Test using offsets"
    (with-bindings {#'dtype/*error-on-slow-path* true}
      (resource/with-resource-context
        (let [test-data [1 2 3 243 244 245]
              short-buf (short-array test-data)
              test-buf (tvm/make-cpu-device-buffer :uint8 6)
              end-buf (double-array 3)]
          (dtype/copy! short-buf 3 test-buf 3 3)
          (dtype/copy! test-buf 3 end-buf 0 3)
          (is (= (->> (drop 3 test-data) (map unchecked-short) vec)
                 (mapv int end-buf))))))))


(deftest host-sub-buffer
  (testing "Sub buffer works"
    (resource/with-resource-context
      (let [test-data [1 2 3 4 5 6]
            short-buf (short-array test-data)
            test-buf (tvm/make-cpu-device-buffer :float32 6)
            view-buf (drv/sub-buffer test-buf 3 3)
            end-buf (double-array 6)]
        (dtype/copy! short-buf 0 test-buf 0 6)
        (dtype/copy! short-buf 0 view-buf 0 3)
        (dtype/copy! test-buf 0 end-buf 0 6)
        (is (= [1 2 3 1 2 3]
               (mapv int end-buf)))))))
