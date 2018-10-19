(ns tech.compute.tvm
  (:require [tech.compute.tvm.driver :as tvm-driver]
            [tech.compute :as compute]
            [tvm-clj.api :as tvm-api]
            [tvm-clj.tvm-bindings :as bindings]
            [tech.compute.tvm.registry :as tvm-reg]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn enable-cpu-driver!
  []
  (require '[tech.compute.tvm.cpu]))


(defn enable-gpu-drivers!
  []
  (require '[tech.compute.tvm.gpu]))


(defn device-type
  "Generically get the tvm device type from a thing"
  [item]
  (-> item
      compute/->driver
      tvm-driver/device-type))


(defn device-id
  "Generically get the tvm device id from a thing."
  [item]
  (-> item
      compute/->device
      tvm-driver/device-id))


(defn cpu?
  [item]
  (= :cpu (device-type item)))


(defn device-type->driver
  [device-type]
  (tvm-reg/device-type->driver device-type))


(defn device-id->device
  [driver device-id]
  (tvm-driver/device-id->device driver device-id))


(defn device-type-id->device
  [device-type & [device-id]]
  (-> (device-type->driver device-type)
      (device-id->device (or device-id 0))))

(defn gpu-scheduling?
  [item]
  (tvm-driver/gpu-scheduling? item))


;;https://github.com/dmlc/tvm/issues/984
(defn scalar-datatype->device-datatype
  [driver scalar-datatype]
  (tvm-driver/scalar-datatype->device-datatype driver scalar-datatype))


(defn schedule-injective!
  "A large class of functions are injective, meaning that they are result-element
  by result-element parallelizeable.  Thus fusing all axis and then running things
  in parallel works at least passably well."
  [driver schedule compute-op-or-vec options]
  (tvm-driver/schedule-injective! driver schedule compute-op-or-vec options))

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
  (tvm-driver/->module driver sched-data-seq
                       (merge {:build-config (merge tvm-api/default-build-config
                                                    build-config)
                               :target-host target-host}
                              (dissoc opts :build-config))))


(defn enumerate-device-ids
  [device-type]
  (let [device-type-int (bindings/device-type->device-type-int device-type)]
    (->> (range)
         (take-while #(= 1 (bindings/device-exists? device-type-int %))))))


(defn make-cpu-device-buffer
  "Make a cpu device buffer.  These are used as host buffers for all devices and device
  buffers for the cpu device."
  [datatype elem-count]
  (when-not (resolve 'tech.compute.tvm.cpu/driver)
    (require 'tech.compute.tvm.cpu))
  (let [device (-> (device-type->driver :cpu)
                   (device-id->device 0))]
    (compute/allocate-device-buffer device elem-count datatype)))


(defn schedule->fn
  [driver {:keys [schedule name arglist bind-map] :as schedule-data} & args]
  (let [{:keys [module fn-map]} (apply ->module driver [schedule-data] args)]
    (get fn-map name)))


(defn has-byte-offset?
  "Used for code generation, because some tvm driver buffer types do not
  support pointer offsetting, you pass in a pointer + byte offset when doing'
  code generation."
  [buffer]
  (tvm-driver/has-byte-offset? buffer))


(defn call-function
  "Call a tvm function on this stream with these arguments."
  [stream fn & args]
  (tvm-driver/call-function stream fn args))
