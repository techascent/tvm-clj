(defproject tvm-clj "0.1.0-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [thinktopic/think.resource "1.2.1"]
                 [techascent/tech.javacpp-datatype "0.2.0"]
                 [thinktopic/think.parallel "0.3.8"]
                 [potemkin "0.4.4"]]
  :java-source-paths ["java"]
  :native-path "java/native/"
  :aot [tvm-clj.compile]
  :main tvm-clj.compile)
