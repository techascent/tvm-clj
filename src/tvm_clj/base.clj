(ns tvm-clj.base
  "Base types for the tvm-clj system.  This avoids the issue where a recompilation leaves
  protocols in difficult-to-understand states or redefines records.  Nothing should be defined
  in this file but types; this allows repl recompilation to succeed predictably"
  (:require [potemkin :as p])
  (:import [tvm_clj.tvm runtime runtime$TVMFunctionHandle runtime$TVMValue
            runtime$NodeHandle runtime$TVMModuleHandle
            runtime$DLTensor runtime$TVMStreamHandle]))


(defprotocol PJVMTypeToTVMValue
  "Convert something to a [long tvm-value-type] pair"
  (->tvm-value [jvm-type]))


(defprotocol PToTVM
  "Convert something to some level of tvm type."
  (->tvm [item]))


(extend-protocol PToTVM
  runtime$TVMFunctionHandle
  (->tvm [item] item)
  runtime$TVMValue
  (->tvm [item] item)
  runtime$NodeHandle
  (->tvm [item] item)
  runtime$TVMModuleHandle
  (->tvm [item] item)
  runtime$DLTensor
  (->tvm [item] item)
  runtime$TVMStreamHandle
  (->tvm [item] item))


(defprotocol PConvertToNode
  (->node [item]))


(defrecord ArrayHandle [^runtime$DLTensor tvm-jcpp-handle]
  PToTVM
  (->tvm [_] tvm-jcpp-handle))

(defrecord StreamHandle [^long device ^long dev-id ^runtime$TVMStreamHandle tvm-hdl]
  PToTVM
  (->tvm [_] tvm-hdl))
