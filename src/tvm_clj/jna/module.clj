(ns tvm-clj.jna.module
  (:require [tvm-clj.jna.base :as jna-base]
            [tech.v3.jna :refer [checknil] :as jna]
            [tvm-clj.bindings.protocols :as bindings-proto]
            [tvm-clj.jna.fns.runtime :as runtime]
            [tech.v3.resource :as resource])
  (:import [com.sun.jna Native NativeLibrary Pointer Function Platform]
           [com.sun.jna.ptr PointerByReference IntByReference LongByReference]
           [tvm_clj.tvm DLPack$DLContext DLPack$DLTensor DLPack$DLDataType
            DLPack$DLManagedTensor]))


(jna-base/make-tvm-jna-fn TVMModFree
                          "Free a module"
                          Integer
                          [module checknil])


(defrecord ModuleHandle [^Pointer tvm-hdl]
  bindings-proto/PToTVM
  (->tvm [item] item)
  bindings-proto/PJVMTypeToTVMValue
  (->tvm-value [item] [(Pointer/nativeValue tvm-hdl) :module-handle])
  jna/PToPtr
  (->ptr-backing-store [item] tvm-hdl))


(defmethod jna-base/tvm-value->jvm :module-handle
  [long-val val-type-kwd]
  (-> (->ModuleHandle (Pointer. long-val))
      (resource/track {:dispose-fn #(TVMModFree (Pointer. long-val))
                       :track-type :auto})))


(jna-base/make-tvm-jna-fn TVMFuncFree
                          "Free a tvm module function"
                          Integer
                          [handle checknil])


(defrecord ModuleFunctionHandle [^Pointer handle]
  bindings-proto/PToTVM
  (->tvm [item] item)
  jna/PToPtr
  (->ptr-backing-store [item] handle))


(jna-base/make-tvm-jna-fn TVMModGetFunction
                          "Get module function"
                          Integer
                          [mod checknil]
                          [func_name jna/string->ptr]
                          [query_imports int]
                          [out jna-base/ptr-ptr])


(defn get-module-function
  [module ^String fn-name query-imports?]
  (let [retval (PointerByReference.)]
    (jna-base/check-call (TVMModGetFunction module fn-name (int (if query-imports? 1 0)) retval))
    (when (= 0 (Pointer/nativeValue (.getValue retval)))
      (throw (ex-info "Could not find module function"
                      {:fn-name fn-name})))
    (resource/track (->ModuleFunctionHandle (.getValue retval))
                    {:dispose-fn #(TVMFuncFree (.getValue retval))
                     :track-type :auto})))


(defn get-module-source
  [module format]
  (runtime/ModuleGetSource module format))


(jna-base/make-tvm-jna-fn TVMModImport
                          "Import one module into another"
                          Integer
                          [mod checknil]
                          [dep checknil])


(defn mod-import
  [mod dep]
  (jna-base/check-call (TVMModImport mod dep)))
