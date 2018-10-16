(ns tvm-clj.compute.registry
  "Additional protocols for tvm drivers, devices, and streams.
Centralized registring of drivers allowing a symbolic name->driver table."
  (:require [tvm-clj.tvm-bindings :as bindings]
            [tvm-clj.api :as tvm-api]))

(defn cpu-device-type
  ^long []
  (bindings/device-type->device-type-int :cpu))

(defn cuda-device-type
  ^long []
  (bindings/device-type->device-type-int :cuda))

(defn opencl-device-type
  ^long []
  (bindings/device-type->device-type-int :opencl))

(defn rocm-device-type
  ^long []
  (bindings/device-type->device-type-int :rocm))


(defprotocol PDeviceInfo
  (device-id [item]
    "Return the tvm integer device of a given device, buffer or stream."))


(defprotocol PDriverInfo
  (device-type [item]
    "Return the tvm integer device type of a given driver, device, buffer, or stream."))


(defprotocol PDeviceIdToDevice
  (device-id->device [driver device-id]))

;;Mapping from integer device types to drivers implementing that type.
(defonce ^:dynamic *device-types->drivers* (atom {}))


(defn add-device-type
  [^long device-type driver]
  (swap! *device-types->drivers* assoc device-type driver))


(defn get-driver
  [device-type]
  (let [device-type (long (if (number? device-type)
                            device-type
                            (bindings/device-type->device-type-int device-type)))]
    (if-let [driver (get @*device-types->drivers* device-type)]
      driver
      (throw (ex-info "Failed to find driver for device type:"
                      {:device-type device-type})))))


(defn get-device
  [device-type ^long device-id]
  (-> (get-driver device-type)
      (device-id->device device-id)))


(defn device-type-int->keyword
  [device-type-int]
  (bindings/device-type-int->device-type device-type-int))


(defn device-type-kwd
  [device]
  (-> (device-type device)
      device-type-int->keyword))


(defprotocol PCompileModule
  (gpu-scheduling? [driver])
  ;;https://github.com/dmlc/tvm/issues/984
  (device-datatypes? [driver])
  ;;Basic injective scheduling.  Updates stage
  (schedule-injective! [driver stage compute-op options])
  ;;Build the module.  See api/schedules->fns
  (->module-impl [driver sched-data-seq options]))


(defn ->module
  "Given a sequence of schedule-data, return a map of name to clojure
  callable function.
  A module is created and added to the resource context transparently.
  Schedule data:
  {:name :fn-name
   :arglist arguments
   :schedule schedule
   :bind-map (optional) bind-map
  }
  returns:
  {:module module
   :fn-map map of name->IFn (clojure callable function.)"
  [driver sched-data-seq & {:keys [build-config target-host]
                            :or {build-config tvm-api/default-build-config
                                 target-host :llvm}
                            :as opts}]

  (->module-impl driver sched-data-seq (merge {:build-config (merge tvm-api/default-build-config
                                                                    build-config)
                                               :target-host target-host}
                                              (dissoc opts :build-config))))


(defn schedule->fn
  [driver {:keys [schedule name arglist bind-map] :as schedule-data} & args]
  (let [{:keys [module fn-map]} (apply ->module driver [schedule-data] args)]
    (get fn-map name)))


(defprotocol PTVMStream
  (call-function-impl [stream fn arg-list]))


(defn call-function
  [stream fn & args]
  (call-function-impl stream fn args))
