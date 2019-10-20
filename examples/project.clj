(defproject tvm-example "0.1.0-SNAPSHOT"
  :description "Example project using tvm"
  :url "http://github.com/tech-ascent/tvm-clj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [tvm-clj "4.7-SNAPSHOT"]
                 [techascent/tech.opencv "4.25"]]
  ;;This is useful if you want to see where the loaded tvm library
  ;;is coming from.  We really recommend that you install a tvm
  ;;built specifically for your system into /usr/lib, however, as there
  ;;are quite a few options possible for tvm.
  :jvm-opts ["-Djna.debug_load=true"]
  )
