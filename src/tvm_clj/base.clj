(ns tvm-clj.base
  "Base types for the tvm-clj system.  This avoids the issue where a recompilation leaves
  protocols in difficult-to-understand states or redefines records.  Nothing should be defined
  in this file but types; this allows repl recompilation to succeed predictably"
  (:import [tvm_clj.tvm runtime runtime$TVMFunctionHandle runtime$TVMValue
            runtime$NodeHandle]))


(defprotocol PJVMTypeToTVMValue
  (jvm->tvm-value [jvm-type]))


(defrecord NodeHandle [^runtime$NodeHandle tvm-jcpp-handle])


(defprotocol PConvertToNode
  (->node [item]))
