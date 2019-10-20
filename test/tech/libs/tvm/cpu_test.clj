(ns tech.libs.tvm.cpu-test
  (:require [clojure.test :refer :all]
            [tech.v2.datatype :as dtype]
            [tech.v2.datatype.functional :as dfn]
            [tech.resource :as resource]
            [tech.compute :as compute]
            [tech.compute.context :as compute-ctx]
            [tech.libs.tvm :as tvm]
            [tvm-clj.api-test :as api-test]
            [tech.libs.tvm.cpu]))


(defn test-add-fn
  [device-type]
  (resource/stack-resource-context
   (compute-ctx/with-context
     {:driver (tvm/driver device-type)}
     (let [device (compute-ctx/default-device)
           stream (compute-ctx/default-stream)
           add-fn (api-test/create-myadd-fn device-type)
           test-data (range 10)
           driver (compute/->driver device)
           host-buf (compute/allocate-host-buffer driver 10 :float32)
           dev-buf-a (compute/allocate-device-buffer device 10 :float32)
           dev-buf-b (compute/allocate-device-buffer device 10 :float32)
           dev-buf-c (compute/allocate-device-buffer device 10 :float32)
           result (int-array 10)]
       (dtype/copy-raw->item! test-data host-buf 0 {})
       (compute/copy-device->device host-buf 0 dev-buf-a 0 10)
       (compute/copy-device->device host-buf 0 dev-buf-b 0 10)
       (tvm/call-function stream add-fn dev-buf-a dev-buf-b dev-buf-c)
       (compute/copy-device->device dev-buf-c 0 host-buf 0 10)
       (compute/sync-with-host stream)
       (dtype/copy! host-buf 0 result 0 10)
       (is (dfn/equals (dfn/+ test-data test-data)
                       (vec result)))))))


(deftest cpu-basic-add
  (test-add-fn :cpu))
