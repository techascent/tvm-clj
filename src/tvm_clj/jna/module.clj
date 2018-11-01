(ns tvm-clj.jna.module
  (:require [tvm-clj.jna.base :refer [make-tvm-jna-fn
                                      device-type->int
                                      device-id->int
                                      ptr-ptr
                                      check-call
                                      ->long-ptr
                                      datatype->dl-datatype
                                      global-function
                                      tvm-value->jvm]
             :as tvm-jna-base]
            [tech.jna :refer [checknil] :as jna]
            [tvm-clj.bindings.protocols :refer [->tvm
                                                base-ptr
                                                ->tvm-value
                                                byte-offset] :as bindings-proto]
            [tech.resource :as resource]
            [tech.datatype.jna :as dtype-jna]
            )
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]))


(make-tvm-jna-fn TVMModFree
                 "Free a module"
                 Integer
                 [module checknil])


(defrecord ModuleHandle [^Pointer tvm-hdl]
  bindings-proto/PToTVM
  (->tvm [item] item)
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [item] [(Pointer/nativeValue tvm-hdl) :module-handle])
  dtype-jna/PToPtr
  (->ptr-backing-store [item] tvm-hdl)
  resource/PResource
  (release-resource [item]
    (check-call (TVMModFree item))))


(defmethod tvm-value->jvm :module-handle
  [long-val val-type-kwd]
  (-> (->ModuleHandle (Pointer. long-val))
      resource/track))


(make-tvm-jna-fn TVMFuncFree
                 "Free a tvm module function"
                 Integer
                 [handle checknil])


(defrecord ModuleFunctionHandle [^Pointer handle]
  bindings-proto/PToTVM
  (->tvm [item] item)
  dtype-jna/PToPtr
  (->ptr-backing-store [item] handle)
  resource/PResource
  (release-resource [item]
    (check-call (TVMFuncFree handle))))


(make-tvm-jna-fn TVMModGetFunction
                 "Get module function"
                 Integer
                 [mod checknil]
                 [func_name jna/string->ptr]
                 [query_imports int]
                 [out ptr-ptr])


(defn get-module-function
  [module ^String fn-name query-imports?]
  (let [retval (PointerByReference.)]
    (check-call (TVMModGetFunction module fn-name (int (if query-imports? 1 0)) retval))
    (when (= 0 (Pointer/nativeValue (.getValue retval)))
      (throw (ex-info "Could not find module function"
                      {:fn-name fn-name})))
    (resource/track (->ModuleFunctionHandle (.getValue retval)))))


(defn get-module-source
  [module format]
  (global-function "module._GetSource" module format))


(make-tvm-jna-fn TVMModImport
                 "Import one module into another"
                 Integer
                 [mod checknil]
                 [dep checknil])


(defn mod-import
  [mod dep]
  (check-call (TVMModImport mod dep)))
