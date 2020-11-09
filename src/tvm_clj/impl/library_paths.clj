(ns tvm-clj.jna.library-paths
  (:require [tech.v3.jna :as jna])
  (:import [java.io File]))


(def tvm-library-name "tvm")


;;Setup library search paths
;;Add the full path to the development system
(jna/add-library-path tvm-library-name :system
                      (str (System/getProperty "user.dir")
                           File/separator
                           "incubator-tvm/build"
                           File/separator
                           (jna/map-shared-library-name tvm-library-name)))

(when-let [tvm-home (System/getenv "TVM_HOME")]
  (when (.exists (File. tvm-home))
    (jna/add-library-path tvm-library-name :system
                          (str tvm-home
                               File/separator
                               "build"
                               File/separator
                               (jna/map-shared-library-name tvm-library-name)))))

(jna/add-library-path tvm-library-name :system tvm-library-name)
;;Then if nothing else works use the packaged library
;;that only supports a couple things if any and may not load.
(jna/add-library-path tvm-library-name :java-library-path tvm-library-name)
