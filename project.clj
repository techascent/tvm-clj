(defproject tvm-clj "0.1.6"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.compute "0.6.8"]
                 [techascent/tech.javacpp-datatype "0.5.8"]
                 [potemkin "0.4.4"]]

  :profiles {:dev
             ;;Unit tests need this.
             {:dependencies [[techascent/tech.opencv "0.2.7"]]}}

  :java-source-paths ["java"]
  :native-path "java/native/"
  :aot [tvm-clj.jni]
  :test-selectors {:default (complement :cuda)
                   :cuda :cuda}

  :aliases {"jni" ["run" "-m" "tvm-clj.jni"]})
