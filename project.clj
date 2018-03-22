(defproject tvm-clj "0.1.0-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.compute "0.5.2"]
                 [potemkin "0.4.4"]]

  :profiles {:dev
             ;; there are a set of small functions that aren't compiled into the javacpp library but into each
             ;; presets library.  So in order to test or do development we have to load one of the presets libraries;
             ;; any one that uses javacpp will do
             {:dependencies [[org.bytedeco.javacpp-presets/opencv-platform "3.4.0-1.4"]]}}

  :java-source-paths ["java"]
  :native-path "java/native/"
  :aot [tvm-clj.compile]
  :main tvm-clj.compile)
