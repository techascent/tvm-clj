(defproject tvm-clj "0.1.0-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [ml.dmlc.tvm/tvm4j-full-linux-x86_64-gpu "0.0.1-SNAPSHOT"]
                 [thinktopic/think.resource "1.2.1"]
                 [thinktopic/think.datatype "0.3.17"]
                 [thinktopic/think.parallel "0.3.8"]])
