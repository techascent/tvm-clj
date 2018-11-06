(ns tvm-clj.jna.library-paths
  (:require [tech.jna :as jna])
  (:import [java.io File]))


(def tvm-library-name "tvm")


;;Setup library search paths
;;Add the full path to the development system
(jna/add-library-path tvm-library-name :system (str (System/getProperty "user.dir")
                                              File/separator
                                              "tvm/build"
                                              File/separator
                                              (jna/map-shared-library-name tvm-library-name)))
(jna/add-library-path tvm-library-name :system tvm-library-name)
;;Then if nothing else works use the packaged library
;;that only supports a couple things if any and may not load.
(jna/add-library-path tvm-library-name :java-library-path tvm-library-name)
