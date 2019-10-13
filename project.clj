(defproject tvm-clj "4.7-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [techascent/tech.compute "4.22"]
                 [potemkin "0.4.4"]]

  :profiles {:dev
             ;;Unit tests need this.
             {:dependencies [[techascent/tech.opencv "4.22"]
                             [cnuernber/libpython-clj "1.5"]]}}

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
