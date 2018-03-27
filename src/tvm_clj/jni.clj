(ns tvm-clj.jni
  (:gen-class)
  (:import [org.bytedeco.javacpp.tools Builder]))


(defn build-java-stub
  []
  (Builder/main (into-array String ["tvm_clj.tvm.presets.runtime" "-d" "java"])))


(defn build-jni-lib
  []
  (Builder/main (into-array String ["tvm_clj.tvm.runtime" "-d"
                                    (str (System/getProperty "user.dir")
                                         "/java/native/linux/x86_64")
                                    "-nodelete" ;;When shit doesn't work this is very helpful
                                    "-Xcompiler"
                                    (str "-I" (System/getProperty "user.dir") "/tvm/include/tvm")
                                    "-Xcompiler"
                                    (str "-I" (System/getProperty "user.dir") "/tvm/dlpack/include")
                                    "-Xcompiler"
                                    "-std=c++11"
                                    "-Xcompiler"
                                    (str "-Wl," "--no-as-needed")
                                    "-Xcompiler"
                                    (str "-Wl," "-ltvm_topi")
                                    "-Xcompiler"
                                    (str "-L" (System/getProperty "user.dir") "/tvm/lib")])))


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
      (build-jni-lib))))
