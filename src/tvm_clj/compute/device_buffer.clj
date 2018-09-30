(ns tvm-clj.compute.device-buffer
  (:require [tvm-clj.core :as tvm-core]
            [tvm-clj.base :as tvm-base]
            [tvm-clj.compute.registry :as tvm-reg]
            [tech.compute.driver :as drv]
            [tech.datatype.base :as dtype]
            [tvm-clj.compute.typed-pointer :as hbuf]
            [tech.typed-pointer :as typed-pointer]
            [think.resource.core :as resource]
            [clojure.core.matrix.protocols :as mp]
            [tech.javacpp-datatype :as jcpp-dtype]
            [tech.datatype.marshal :as marshal]
            [tech.compute.tensor :as ct])
  (:import [tvm_clj.tvm runtime$DLTensor runtime runtime$DLContext]
           [tvm_clj.base ArrayHandle]
           [org.bytedeco.javacpp Pointer LongPointer]
           [java.lang.reflect Field]
           [tech.compute.tensor Tensor]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn base-ptr-dtype
  [datatype]
  (if (typed-pointer/signed-datatype? datatype)
    datatype
    (typed-pointer/direct-conversion-map datatype)))

(defn tvm-ary->pointer
  ^Pointer [^ArrayHandle ten-ary ^long elem-count datatype]
  (let [^runtime$DLTensor tensor (.tvm-jcpp-handle ten-ary)
        tens-ptr (.data tensor)
        ptr-dtype (base-ptr-dtype datatype)
        retval (jcpp-dtype/make-empty-pointer-of-type ptr-dtype)]
    (.set ^Field jcpp-dtype/address-field retval (+ (.address tens-ptr)
                                                    (.byte_offset tensor)))
    (jcpp-dtype/set-pointer-limit-and-capacity retval elem-count)))



(defn is-cpu-device?
  [device]
  (= runtime/kDLCPU (tvm-reg/device-type device)))


(declare device-buffer->ptr)


(defrecord DeviceBuffer [device dev-ary]
  dtype/PDatatype
  (get-datatype [buf] (:datatype dev-ary))

  mp/PElementCount
  (element-count [buf] (apply * (:shape dev-ary)))

  drv/PBuffer
  (sub-buffer-impl [buffer offset length]
    (let [^runtime$DLTensor tvm-tensor (tvm-base/->tvm dev-ary)
          base-ptr (.data tvm-tensor)
          datatype (dtype/get-datatype buffer)]
      (->DeviceBuffer device
                      (tvm-base/pointer->tvm-ary
                       base-ptr
                       (tvm-reg/device-type device)
                       (tvm-reg/device-id device)
                       datatype
                       [length]
                       nil
                       ;;add the byte offset where the new pointer should start
                       (* (long offset) (long (dtype/datatype->byte-size
                                               datatype)))))))
  (alias? [lhs rhs]
    (hbuf/jcpp-pointer-alias? (tvm-ary->pointer dev-ary)
                              (:dev-ary rhs)))
  (partially-alias? [lhs rhs]
    (hbuf/jcpp-pointer-partial-alias? (tvm-ary->pointer dev-ary
                                                        (mp/element-count lhs)
                                                        (:datatype dev-ary))
                                      (tvm-ary->pointer (:dev-ary rhs)
                                                        (mp/element-count rhs)
                                                        (:datatype dev-ary))))

  tvm-base/PToTVM
  (->tvm [item] dev-ary)


  tvm-base/PJVMTypeToTVMValue
  (->tvm-value [_]
    (tvm-base/->tvm-value dev-ary))


  dtype/PAccess
  (set-value! [item offset value]
    (dtype/set-value! (typed-pointer/->typed-pointer item) offset value))
  (set-constant! [item offset value elem-count]
    (dtype/set-constant! (typed-pointer/->typed-pointer item) offset value elem-count))
  (get-value [item offset]
    (dtype/get-value (typed-pointer/->typed-pointer item) offset))

  typed-pointer/PToPtr
  (->ptr [item] (device-buffer->ptr item))

  ;;Efficient bulk copy is provided by this line and implementing the PToPtr protocol
  marshal/PContainerType
  (container-type [this] :typed-pointer)

  ;;The underlying tvm array is tracked by the system so there is no
  ;;need to release this resource.
  resource/PResource
  (release-resource [_] )

  tvm-reg/PDeviceInfo
  (device-id [_] (tvm-reg/device-id device))

  tvm-reg/PDriverInfo
  (device-type [_] (tvm-reg/device-type device))

  drv/PDeviceProvider
  (get-device [_] device))


(defn device-buffer->ptr
  "Get a javacpp pointer from a device buffer.  Throws if this isn't a cpu buffer"
  [^DeviceBuffer buffer]
  (when-not (is-cpu-device? (.device buffer))
    (throw (ex-info "Can only get pointers from cpu device buffers"
                    {})))
  (tvm-ary->pointer (.dev-ary buffer) (mp/element-count buffer) (dtype/get-datatype buffer)))


(defn make-device-buffer-of-type
  [device datatype elem-count]
  (resource/track
   (->> (tvm-core/allocate-device-array [elem-count] datatype
                                        (tvm-reg/device-type device)
                                        (tvm-reg/device-id device))
        (->DeviceBuffer device))))


(defn make-cpu-device-buffer
  "Make a cpu device buffer.  These are used as host buffers for all devices
and device buffers for the cpu device."
  [datatype elem-count]
  (when-not (resolve 'tvm-clj.compute.cpu/driver)
    (require 'tvm-clj.compute.cpu))
  (make-device-buffer-of-type (tvm-reg/get-device runtime/kDLCPU 0)
                              datatype
                              elem-count))


(defn device-buffer->tvm-array
  ^runtime$DLTensor [^DeviceBuffer buf]
  (tvm-base/->tvm (tvm-base/->tvm buf)))


(defn has-byte-offset?
  [tensor]
  (let [buf (ct/tensor->buffer tensor)
        ^ArrayHandle buf-data (tvm-base/->tvm buf)
        ^runtime$DLTensor backing-store (tvm-base/->tvm buf-data)]
    (not= 0 (.byte_offset backing-store))))


(extend-type Tensor
  tvm-base/PJVMTypeToTVMValue
  (->tvm-value [item]
    (let [^runtime$DLTensor src-dl-tensor
          (device-buffer->tvm-array (ct/tensor->buffer item))
          ^runtime$DLContext ctx (.ctx src-dl-tensor)
          dims (ct/tensor->dimensions item)
          stride-data (when-not (ct/dense? item)
                        (:strides dims))]
      (-> (tvm-base/pointer->tvm-ary (.data src-dl-tensor)
                                     (.device_type ctx)
                                     (.device_id ctx)
                                     (ct/get-datatype item)
                                     (:shape dims)
                                     stride-data
                                     (.byte_offset src-dl-tensor))
          tvm-base/->tvm-value))))
