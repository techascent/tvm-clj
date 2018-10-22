(ns tech.compute.tvm.registry
  (:require [tech.compute.registry :as registry]
            [tvm-clj.tvm-bindings :as bindings]
            [tech.compute.tvm.driver :as tvm-driver]
            [tech.compute.driver :as drv]
            [tech.compute :as compute]
            [clojure.set :as c-set]))


(defn tvm-driver-name
  [tvm-device-type-kwd]
  (-> (str "tech.compute.tvm." (name tvm-device-type-kwd))
      keyword))


(def ^:dynamic *driver-name->tvm-device-type* (atom {}))


(defn- add-driver-name
  [driver-name tvm-device-type-kwd]
  (swap! *driver-name->tvm-device-type*
         assoc driver-name tvm-device-type-kwd))


(defn driver-name->tvm-device-type
  [driver-name]
  (if-let [retval (get @*driver-name->tvm-device-type* driver-name)]
    retval
    (throw (ex-info "Failed to find tvm device type"
                    {:driver-name driver-name}))))


(defn tvm-device-type->driver-name
  [tvm-device-type]
  (if-let [retval (get (c-set/map-invert @*driver-name->tvm-device-type*)
                       tvm-device-type)]
    retval
    (throw (ex-info "Failed to find device name for device type"
                    {:tvm-device-type tvm-device-type}))))


(defn register-driver
  [driver]
  (registry/register-driver driver)
  (add-driver-name (drv/driver-name driver)
                   (tvm-driver/device-type driver)))


(defn device-type->driver
  [tvm-device-type]
  (-> (tvm-device-type->driver-name tvm-device-type)
      compute/driver))
