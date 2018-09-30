(defproject tvm-clj "0.1.0-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.compute "0.5.6"]
                 [techascent/tech.javacpp-datatype "0.3.5"]
                 [potemkin "0.4.4"]]

  :profiles {:dev
             ;;Unit tests need this.
             {:dependencies [[org.bytedeco.javacpp-presets/opencv-platform "3.4.0-1.4"]]}}

  :java-source-paths ["java"]
  :native-path "java/native/"
  :aot [tvm-clj.jni]
  :test-selectors {:default (complement :cuda)
                   :cuda :cuda}

  :aliases {"jni" ["run" "-m" "tvm-clj.jni"]})
