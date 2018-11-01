(ns tvm-clj.jni
  (:gen-class)
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [tech.jna :as jna])
  (:import [com.sun.jna Platform]))


(defn lib-filename [libname] (jna/map-shared-library-name libname))


(defn filepath [& parts] (string/join (java.io.File/separator) parts))


(defn relpath
  [& parts]
  (apply filepath (System/getProperty "user.dir") parts))


(defn native-path []
  ;;Use the jna subsystem for this type of thing
  (relpath "resources" Platform/RESOURCE_PREFIX))


(defn install-tvm-libs
  [& [libs]]
  (let [libs (or libs ["tvm" "tvm_topi"])]
    (.mkdirs (io/file (native-path)))
    (->> libs
         (map #(io/copy (io/file (relpath "tvm" "build" (lib-filename %)))
                        (io/file (filepath (native-path) (lib-filename %)))))
         doall)))


(defn -main
  [& args]
  (let [arg-val (first args)
        command (if arg-val
                  (keyword arg-val)
                  :build-jni-java)]
    (condp = command
      :install-tvm-libs
      (install-tvm-libs))))
