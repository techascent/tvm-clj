(ns tvm-clj.impl.tvm-ns-fns
  (:require [tvm-clj.impl.base :as base]))


(defn- fn-prefix
  [^String fn-name]
  (let [last-idx (.lastIndexOf fn-name ".")]
    (if (> last-idx 0)
      (.substring fn-name 0 last-idx)
      "")))


(defn safe-local-name
  [^String lname]
  (cond
    (= "String" lname) "RuntimeString"
    :else lname))


(defn fn-postfix
  [^String fn-name]
  (let [last-idx (.lastIndexOf fn-name ".")]
    (-> (if (> last-idx 0)
          (.substring fn-name (inc last-idx))
          "")
        (safe-local-name))))


(defn- fns-with-prefix
  [prefix]
  (->> (base/global-function-names)
       (filter #(= prefix (fn-prefix %)))))


(defmacro export-tvm-functions
  [prefix]
  `(do
     ~@(->> (fns-with-prefix prefix)
            (mapcat (fn [fn-name]
                      (let [local-name (fn-postfix fn-name)
                            global-sym (symbol (str local-name "-fnptr*"))]
                        [`(defonce ~global-sym (delay (base/name->global-function
                                                       ~fn-name)))
                         `(defn ~(symbol local-name)
                            "TVM exported fn"
                            [& ~'args]
                            (with-bindings {#'base/fn-name ~fn-name}
                              (apply base/call-function @~global-sym ~'args)))]))))))
