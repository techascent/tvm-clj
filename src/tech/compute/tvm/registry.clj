(ns tech.compute.tvm.registry
  (:require [tech.compute.registry :as registry]
            [clojure.set :as c-set]))


(defn tvm-driver-name
  [tvm-device-type-kwd]
  `(-> (str "tech.compute.tvm." (name ~tvm-device-type-kwd))
       keyword))


(def ^:dynamic *driver-name->tvm-device-type-kwd* (atom {}))


(defn add-driver
  [driver-name tvm-device-type-kwd]
  (swap! *driver-name->tvm-device-type-kwd*
         assoc driver-name tvm-device-type-kwd))


(defn driver-name->tvm-device-type-kwd
  [driver-name]
  (if-let [retval (get @*driver-name->tvm-device-type-kwd* driver-name)]
    retval
    (throw (ex-info "Failed to find tvm device type"
                    {:driver-name driver-name}))))


(defn tvm-device-type-kwd->driver-name
  [tvm-device-type-kwd]

  )
