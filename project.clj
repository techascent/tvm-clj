(defproject tvm-clj "5.1-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [techascent/tech.datatype "5.0-alpha-1"]
                 [potemkin "0.4.4"]]

  :profiles {:dev
             ;;Unit tests need this.
             {:dependencies [[techascent/tech.opencv "5.0-alpha-1"]]}}

  :java-source-paths ["java"]
  :native-path "java/native"

  :clean-targets
  ^{:protect false} [:target-path :compile-path]
  :aot [tvm-clj.jni]
  :test-selectors {:default (complement :cuda)
                   :cuda :cuda}
  :jar {:prep-tasks ["compile" ["jni" "install-tvm-libs"]
                     "compile" ["javac"]]}

  :aliases {"jni" ["run" "-m" "tvm-clj.jni"]})
