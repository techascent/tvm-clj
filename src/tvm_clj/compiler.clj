(ns tvm-clj.compiler
  (:require [tvm-clj.compiler.graph :as g]
            [tvm-clj.compiler.ast :as ast]
            [tvm-clj.compiler.codegen :as codegen]))

(defn compute-graph
  []
  (g/compute-graph))


(defn make-variable
  [graph varname & {:keys [dtype]
                    :or {dtype :int32}}]
  (ast/make-variable graph varname :dtype dtype))


(defn make-tensor-and-buffer
  [graph varname shape & {:keys [sparse? byte-offset? dtype]
                          :or {sparse? false byte-offset? false
                               dtype :float32}}]
  (ast/make-tensor-and-buffer graph varname shape
                              :sparse? sparse?
                              :byte-offset? byte-offset?
                              :dtype dtype))


(defn get-tensor
  [graph varname]
  (g/get-tensor graph varname))


(defn compile-fn
  [driver graph input-fn & args]
  (apply codegen/compile-fn driver graph input-fn args))
