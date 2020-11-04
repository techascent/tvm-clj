(defproject tvm-clj "5.1-SNAPSHOT"
  :description "Clojure bindings and exploration of the tvm library"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.2-alpha1"]
                 [cnuernber/dtype-next "6.00-alpha-17"]
                 [techascent/tech.jna "4.04"]
                 [potemkin "0.4.4"]]

  :java-source-paths ["java"]

  :clean-targets
  ^{:protect false} [:target-path :compile-path]
  :test-selectors {:default (complement :cuda)
                   :cuda :cuda}

  ;;:aliases {"jni" ["run" "-m" "tvm-clj.jni"]}
  )
