(defproject tvm-clj "6.00-beta-1-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/techascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2-alpha1"]
                 [cnuernber/dtype-next "6.00-beta-3"]
                 [techascent/tech.jna "4.05"]]

  :java-source-paths ["java"]

  :profiles {:dev {:dependencies [[criterium "0.4.5"]]}
             :codox
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
                                   tvm-clj.application.image
                                   tvm-clj.application.kmeans]}}}
  :aliases {"codox" ["with-profile" "codox,dev" "codox"]})
