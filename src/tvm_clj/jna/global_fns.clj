(ns tvm-clj.jna.global-fns
  "Namespace so you can call the global functions by symbol and not by string."
  (:require [tvm-clj.jna.base :as jna-base]))


(defn fns-matching-prefix
  "Return all global function names matching prefix but without a '.' in them"
  [& [prefix]]
  (->> (jna-base/global-function-names)
       (filter #(and (if prefix
                       (.startsWith % prefix)
                       true)
                     (not (.contains (.substring % (count prefix))
                                     "."))))
       (map #(if prefix
               (.substring % (count prefix))
               %))))


(defmacro export-global-fns
  [& [prefix]]
  (let [all-fn-names (fns-matching-prefix prefix)]
    `(do
       ~@(->>
          all-fn-names
          (map
           (fn [fn-name]
             (let [full-name (str prefix fn-name)]
               `(defn ~(symbol fn-name)
                  [& args#]
                  (apply jna-base/global-function ~full-name args#)))))))))
