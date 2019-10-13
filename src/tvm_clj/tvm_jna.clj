(ns tvm-clj.tvm-jna
  (:require [clojure.set :as c-set]
            [tech.v2.datatype :as dtype]
            [tech.jna :refer [checknil] :as jna]
            ;;Standard paths for the tvm library
            [tvm-clj.jna.library-paths :as jna-lib-paths]
            ;;PRotocols independent of specific bindings
            [tvm-clj.bindings.protocols :as bindings-proto]
            ;;Definitions indepdnent of specific bindings
            [tvm-clj.bindings.definitions :as definitions]
            [tvm-clj.jna.stream :as stream]
            [tvm-clj.jna.dl-tensor :as dl-tensor]
            [tvm-clj.jna.base :as tvm-jna-base]
            [tvm-clj.jna.node :as node]
            [tvm-clj.jna.module :as module])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn ->tvm-value
  [item]
  (bindings-proto/->tvm-value item))


(defn ->tvm
  [item]
  (bindings-proto/->tvm item))


(defn ->node
  [item]
  (bindings-proto/->node item))


(defn device-id
  [item]
  (bindings-proto/device-id item))


(defn device-type
  [item]
  (bindings-proto/device-type item))


(defn byte-offset
  [item]
  (bindings-proto/byte-offset item))


(defn base-ptr
  [item]
  (bindings-proto/base-ptr item))


(defn is-expression-node?
  [node]
  (definitions/is-expression-node? node))


(defn device-type->int
  ^long [device-type]
  (tvm-jna-base/device-type->int device-type))


(defn device-type-int->device-type
  [^long device-type]
  (definitions/device-type-int->device-type device-type))



(defn create-stream
  [device-type ^long device-id]
  (stream/create-stream device-type device-id))


(defn sync-stream-with-host
  [stream]
  (stream/sync-stream-with-host stream))


(defn sync-stream-with-stream
  [stream]
  (stream/sync-stream-with-stream stream))


(defn set-current-thread-stream
  [stream]
  (stream/set-current-thread-stream stream))


(defn check-cpu-tensor
  [item]
  (dl-tensor/check-cpu-tensor item))



(defn allocate-device-array
  [shape datatype device-type ^long device-id]
  (dl-tensor/allocate-device-array shape datatype device-type device-id))


(defn copy-to-array!
  [src dest-tensor ^long n-bytes]
  (dl-tensor/copy-to-array! src dest-tensor n-bytes))


(defn copy-from-array!
  [src-tensor ^Pointer dest ^long n-bytes]
  (dl-tensor/copy-from-array! src-tensor dest n-bytes))


(defn copy-array-to-array!
  [src dst stream]
  (dl-tensor/copy-array-to-array! src dst stream))


(defn pointer->tvm-ary
  "Take a pointer value and convert it into a dl-tensor datatype.

strides are optional, can be nil.

Not all backends in TVM can offset their pointer types.  For this reason, tvm arrays
  have a byte_offset member that you can use to make an array not start at the pointer's
  base address."
  [ptr device-type device-id
   datatype shape strides
   byte-offset & [gc-root]]
  (dl-tensor/pointer->tvm-ary ptr device-type device-id
                              datatype shape strides
                              byte-offset gc-root))

(defn call-function
  [tvm-fn & args]
  (apply tvm-jna-base/call-function tvm-fn args))


(defn global-function
  [fn-name & args]
  (apply tvm-jna-base/global-function fn-name args))


(def global-node-function tvm-jna-base/global-function)
(def g-fn tvm-jna-base/global-function)
(def gn-fn tvm-jna-base/global-function)


(defn is-node-handle?
  [item]
  (node/is-node-handle? item))

(defn get-node-type
  [node-handle]
  (node/get-node-type node-handle))

(defn tvm-array->jvm
  [tvm-ary]
  (node/tvm-array->jvm tvm-ary))

(defn tvm-array
  [jvm-ary]
  (node/tvm-array jvm-ary))


(defn device-exists?
  [device-type ^long device-id]
  (if (= device-type :cpu)
    (= device-id 0)
    (= 1
       (g-fn "_GetDeviceAttr" (tvm-jna-base/device-type->int device-type) device-id
             (definitions/device-attribute-map :exists)))))


(defn get-module-function
  [module ^String fn-name & {:keys [query-imports?]}]
  (module/get-module-function module fn-name query-imports?))


(defn get-module-source
  [module {:keys [format]
           :or {format ""}}]
  (module/get-module-source module format))


(defn mod-import
  [mod dep]
  (module/mod-import mod dep))
