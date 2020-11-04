(ns tvm-clj.jna.fns
  "TVM describes a lot of their C api dynamically; you query for the list of global
  function names and they are returned delimited by periods similar to clojure
  namespaces.  We want to scan the list of functions once and dynamically
  create all sub namespaces.  This does mean these namespaces will not
  have documentation at this point."
  (:require [tvm-clj.jna.base :as jna-base]
            [clojure.string :as s]
            [clojure.tools.logging :as log]
            [clojure.java.io :as io]))


(defn safe-local-name
  [^String lname]
  (cond
    (= "String" lname) "RuntimeString"
    :else lname))


(defn define-tvm-fns!
  []
  (let [namespaces (->> (jna-base/global-function-names)
                        (map (fn [gname]
                               (let [parts (s/split gname #"\.")]
                                 [(butlast parts)
                                  {:fullname gname
                                   :local-name (last parts)}])))
                        (group-by first)
                        (map (fn [[k vs]]
                               (if (seq k)
                                 [k (mapv second vs)]
                                 (do
                                   (log/warnf "Skipping non-namespaced symbols %s"
                                              (mapv (comp :fullname second) vs))
                                   nil))))
                        (remove nil?))
        cur-dir (System/getProperty "user.dir")
        root-ns-path (str cur-dir "/src/tvm_clj/jna/fns/")]
    (doseq [[ns-name ns-data] namespaces]
      ;;Auto generating the namespace only gets you dynamic resolution of the
      ;;names.  So we *actually* define the namespace.
      (let [ns-path (str root-ns-path (s/join "/" ns-name) ".clj")
            ns-name (str "tvm-clj.jna.fns." (s/join "." ns-name))
            builder (StringBuilder.)]
        (.append builder (format "(ns %s
  (:require [tvm-clj.jna.base :as jna-base]))

" ns-name))

        (log/debugf "Generating TVM namespace %s" ns-path)
        (doseq [{:keys [local-name fullname]} ns-data]
          (.append builder (format "(let [gfn* (delay (jna-base/name->global-function \"%s\"))]
  (defn %s
   \"TVM PackedFn\"
   [& args]
   (with-bindings {#'jna-base/fn-name \"%s\"}
     (apply jna-base/call-function @gfn* args))))

"
                                   fullname
                                   (safe-local-name local-name)
                                   fullname)))
        (io/make-parents ns-path)
        (spit ns-path (.toString builder))))))


(comment
  ;;Only need to run this when the version of TVM changes.
  ;;In that case *delete* all files under tvm_clj.jna.fns/*
  ;;and then run and check changes.
  (define-tvm-fns!)
  )
