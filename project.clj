(defproject tvm-clj "5.1-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/techascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2-alpha1"]
                 [cnuernber/dtype-next "6.00-alpha-21"]
                 [techascent/tech.jna "4.05"]
                 [potemkin "0.4.4"]]

  :java-source-paths ["java"]

  :profiles {:codox
             {:dependencies [[codox-theme-rdash "0.1.2"]]
              :plugins [[lein-codox "0.10.7"]]
              :codox {:project {:name "tvm-clj"}
                      :metadata {:doc/format :markdown}
                      :themes [:rdash]
                      :source-paths ["src"]
                      :output-path "docs"
                      :doc-paths ["topics"]
                      :source-uri "https://github.com/techascent/tvm-clj/blob/master/{filepath}#L{line}"
                      :namespaces [tvm-clj.ast
                                   tvm-clj.schedule
                                   tvm-clj.compiler
                                   tvm-clj.module
                                   tvm-clj.device
                                   tvm-clj.application.image]}}}
  :aliases {"codox" ["with-profile" "codox,dev" "codox"]})
