(ns tvm-clj.jni
  (:gen-class)
  (:require [clojure.java.io :as io]
            [clojure.string :as string])
  (:import [org.bytedeco.javacpp.tools Builder]))


(defn lib-filename [libname] (System/mapLibraryName libname))


(defn filepath [& parts] (string/join (java.io.File/separator) parts))


(defn relpath
  [& parts]
  (apply filepath (System/getProperty "user.dir") parts))


(defn native-path []
  (relpath "java" "native"
           (string/lower-case (string/replace (System/getProperty "os.name") #"\s+" ""))
           (System/getProperty "os.arch")))


(defn build-java-stub
  []
  (Builder/main (into-array String ["tvm_clj.tvm.presets.runtime" "-d" "java"])))


(defn build-jni-lib
  []
  (Builder/main (into-array String ["tvm_clj.tvm.runtime" "-d" (native-path)
                                    "-nodelete" ;;When shit doesn't work this is very helpful
                                    "-Xcompiler"
                                    (str "-I" (relpath "tvm" "include" "tvm"))
                                    "-Xcompiler"
                                    (str "-I" (relpath "tvm" "3rdparty" "dlpack" "include"))
                                    "-Xcompiler"
                                    "-std=c++11"
                                    ;; This option breaks on OSX/Xcode/clang
                                    ;;"-Xcompiler"
                                    ;;(str "-Wl," "--no-as-needed")
                                    "-Xcompiler"
                                    (str "-Wl," "-ltvm_topi")
                                    "-Xcompiler"
                                    (str "-L" (relpath "tvm" "build"))])))


(defn install-jni-lib
  []
  (.mkdirs (io/file (native-path)))
  (doall (map #(io/copy (io/file (relpath "tvm" "build" (lib-filename %)))
                        (io/file (filepath (native-path) (lib-filename %))))
              ["tvm" "tvm_topi"])))



(defn build-and-install-jni-lib
  []
  (build-jni-lib)
  (install-jni-lib))


(defn -main
  [& args]
  (let [arg-val (first args)
        command (if arg-val
                  (keyword arg-val)
                  :build-jni-java)]
    (condp = command
      :build-jni-java ;;step 1
      (build-java-stub)
      :build-jni
      (build-jni-lib)
      :install-jni
      (install-jni-lib)
      :build-and-install-jni
      (build-and-install-jni-lib))))
