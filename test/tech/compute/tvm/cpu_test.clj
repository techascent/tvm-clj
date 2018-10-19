(ns tech.compute.tvm.cpu-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [tech.datatype.core :as dtype]
            [tech.resource :as resource]
            [tech.compute :as compute]
            [tech.compute.tvm :as tvm]
            [tvm-clj.api-test :as api-test]
            [tech.compute.tvm.cpu]))


(defn test-add-fn
  [device-type]
  (resource/with-resource-context
    (let [device (tvm/device-type-id->device device-type)
          add-fn (api-test/create-myadd-fn device-type)
          test-data (range 10)
          driver (compute/->driver device)
          host-buf (compute/allocate-host-buffer driver 10 :float32)
          dev-buf-a (compute/allocate-device-buffer device 10 :float32)
          dev-buf-b (compute/allocate-device-buffer device 10 :float32)
          dev-buf-c (compute/allocate-device-buffer device 10 :float32)
          result (int-array 10)
          stream (compute/default-stream device)]
      (dtype/copy-raw->item! test-data host-buf 0 {})
      (compute/copy-host->device host-buf 0 dev-buf-a 0 10)
      (compute/copy-host->device host-buf 0 dev-buf-b 0 10)
      (tvm/call-function stream add-fn dev-buf-a dev-buf-b dev-buf-c)
      (compute/copy-device->host dev-buf-c 0 host-buf 0 10)
      (compute/sync-with-host stream)
      (dtype/copy! host-buf 0 result 0 10)
      (is (m/equals (m/add test-data test-data)
                    (vec result))))))


(deftest cpu-basic-add
  (test-add-fn :cpu))
