(ns tvm-clj.compute.cpu-test
  (:require [tvm-clj.compute.cpu]
            [tvm-clj.compute.registry :as tvm-reg]
            [tvm-clj.core :as tvm-core]
            [tvm-clj.api-test :as api-test]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [think.resource.core :as resource]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(defn test-add-fn
  [device add-fn]
  (drv/with-compute-device device
    (let [test-data (range 10)
          driver (drv/get-driver device)
          host-buf (drv/allocate-host-buffer driver 10 :float32)
          dev-buf-a (drv/allocate-device-buffer 10 :float32)
          dev-buf-b (drv/allocate-device-buffer 10 :float32)
          dev-buf-c (drv/allocate-device-buffer 10 :float32)
          stream (drv/default-stream device)
          result (int-array 10)]
      (dtype/copy-raw->item! test-data host-buf 0)
      (drv/copy-host->device stream host-buf 0 dev-buf-a 0 10)
      (drv/copy-host->device stream host-buf 0 dev-buf-b 0 10)
      (tvm-reg/call-function stream add-fn dev-buf-a dev-buf-b dev-buf-c)
      (drv/copy-device->host stream dev-buf-c 0 host-buf 0 10)
      (drv/sync-with-host stream)
      (dtype/copy! host-buf 0 result 0 10)
      (is (m/equals (m/add test-data test-data)
                    (vec result))))))


(deftest cpu-basic-add
  (resource/with-resource-context
    (test-add-fn (tvm-reg/get-device :cpu 0)
                 (api-test/create-myadd-fn :cpu))))
