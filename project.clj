(defproject tvm-clj "1.4-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.compute "1.9"]
                 [potemkin "0.4.4"]]

  :profiles {:dev
             ;;Unit tests need this.
             {:dependencies [[techascent/tech.opencv "1.2"]]}}

  :java-source-paths ["java"]
  :native-path "java/native/"
  :tvm-clj-runtime-path "java/tvm_clj/tvm/runtime.java"
  :jni-path "java/native"
  :clean-targets
  ^{:protect false} [:target-path :compile-path :jni-path :tvm-clj-runtime-path]
  :aot [tvm-clj.jni]
  :test-selectors {:default (complement :cuda)
                   :cuda :cuda}

  :aliases {"jni" ["run" "-m" "tvm-clj.jni"]})
