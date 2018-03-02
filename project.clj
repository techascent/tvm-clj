(defproject tvm-clj "0.1.0-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.compute "0.2.0"]
                 [potemkin "0.4.4"]]
  :java-source-paths ["java"]
  :native-path "java/native/"
  :aot [tvm-clj.compile]
  :main tvm-clj.compile)
