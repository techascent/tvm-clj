(ns tvm-clj.ast
  "TVM's algorithms are first described using an AST tailored towards
  ND-array programming."
  (:require [tvm-clj.impl.protocols :refer [->node] :as bindings]
            [tvm-clj.impl.node :as jna-node]
            [tvm-clj.impl.fns.te :as te-fns]
            [tvm-clj.impl.fns.tir :as tir-fns]
            [tvm-clj.impl.fns.ir :as ir-fns]
            [tech.v3.datatype :as dtype]
            [clojure.string :as s])
  (:refer-clojure :exclude [range cast mod min max]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defmacro when-not-error
  [condition throw-clause]
  `(when-not (do ~condition)
     (throw ~throw-clause)))


(defn ->dtype
  ^String [dtype-or-name]
  (jna-node/->dtype dtype-or-name))

(def ^:dynamic *varname-prefix* "")

(defn safe-str
  [str-name]
  (let [str-name (if (keyword? str-name)
                   (name str-name)
                   (str str-name))]
    (s/replace (str *varname-prefix* str-name) "-" "_")))


(defn variable
  "Create a scalar variable.  Returns a node handle"
  [^String name & {:keys [dtype]
                   :or {dtype "int32"}}]
  (tir-fns/Var (safe-str name) (->dtype dtype)))


(defn placeholder
  "Create a user-supplied tensor variable"
  [shape name & {:keys [dtype]
                 :or {dtype "float32"}}]
  (let [shape (->> (if-not (instance? clojure.lang.Seqable shape)
                     [shape]
                     shape)
                   (mapv ->node))]
    (te-fns/Placeholder shape
                        (->dtype dtype)
                        (safe-str name))))


(defn- range
  "Create a range with defined start inclusive and end exclusive"
  [start end]
  (ir-fns/Range start end))



(defn tget
  "Get an item from a tensor"
  [tensor indices]
  (let [indices (if (number? indices)
                  [indices]
                  indices)]
    (when-not-error (= (count (:shape tensor))
                       (count indices))
      (ex-info "Num indices must match tensor rank"
               {:tensor-range (count (:shape tensor))
                :index-count (count indices)}))
    (let [indices
          (mapv (fn [index-val]
                  (let [node-data (->node index-val)
                        node-type-name (bindings/node-type-name node-data)]
                    (if (= "tir.IterVar" node-type-name)
                      (:var node-data)
                      node-data)))
                indices)]
      (tir-fns/ProducerLoad tensor indices))))


(defmacro tvm-let
  "Lets in tvm must be nested.  This leads to an exciting macro.
  Pairs must be of the form var val-expr.  Body is *not* an implicit
  do!!"
  [expr-pairs body]
  (->> expr-pairs
       (partition 2)
       reverse
       (reduce (fn [data [var-symbol expr]]
                 `(let [evaled-expr# ~expr
                        ~var-symbol (variable ~(safe-str var-symbol)
                                              :dtype (:dtype evaled-expr#))]
                    (bindings/g-fn "make.Let" ~var-symbol evaled-expr# ~data)))
               body)))


(def ^:private iteration-variable-types
  "Iteration variable types defined in tvm/include/Expr.h"
  {
   ;; /*!
   ;; * \brief Data parallel iteration.
   ;; *  This normally corresponds to axis of Tensor.
   ;; *  Allow all IterVar manipulations.
   ;; *
   ;; * \note This does not mean the loop
   ;; *  have to be executed in parallel fashion.
   ;; */
   :data-parallel 0
   ;; /*!
   ;; * \brief The IterVar itself is a thread-index
   ;; *  of a fixed thread launching group.
   ;; *  Note that this is already assumed to be parallelized.
   ;; *
   ;; *  Disallow: split/fuse/vectorize/parallel
   ;; */
   :thread-index 1
   ;; /*!
   ;; * \brief Communicative reduction.
   ;; *  Cannot be directly parallelized.
   ;; *
   ;; *  Disallow: parallel/vectorize
   ;; */
   :communicative-reduce 2
   ;; /*!
   ;; * \brief Serial loops with loop carry dependency,
   ;; *  the iteration must execute in order.
   ;; *  Cannot be re-ordered.
   ;; *
   ;; *  Disallow: reorder/parallel/vectorize
   ;; */
   :ordered 3
   ;; /*!
   ;; * \brief IterVar is opaque,
   ;; *
   ;; *  May not corresponds to any generated loop
   ;; *  Disallow all IterVar manipulations and compute_at
   ;; *
   ;; * \note This is usually used to implement composite op
   ;; *  or external op, where the
   ;; */
   :dim-info 4
   ;; // The following are possible additional
   ;; // types that are provided during schedule
   ;; /*!
   ;; * \brief The execution is unrolled.
   ;; */
   :unrolled 5
   ;; /*!
   ;; * \brief The loop is vectorized.
   ;; */
   :vectorized 6
   ;; /*!
   ;; * \brief The loop is parallelized.
   ;; */
   :parallelized 7
   ;; /*!
   ;; * \brief Marks boundary of tensorization intrinsic.
   ;; */
   :tensorized 8})


(def iteration-variable-type-set (set (keys iteration-variable-types)))


(defn iteration-variable
  "Create a variable that controls iteration through the data.  The iteration type
affects the class of optimizations that the compiler is able to apply to the affected
expressions,

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of iteration variable.

    iteration-type : keyword
        The type of iteration.

    thread-tag : str
        The thread tag of the iteration variable."
  [domain name iteration-type & {:keys [thread-tag]
                                 :or {thread-tag ""}}]

  (when-not-error (iteration-variable-type-set iteration-type)
    (ex-info "Iteration type not in allowed iteration types"
             {:allowed-types iteration-variable-type-set
              :iteration-type iteration-type}))

  (let [domain (when domain
                 (if (and (bindings/is-node-handle? domain)
                          (= :range (bindings/node-type-name domain)))
                   domain
                   (range (first domain) (second domain))))
        v (variable name)]
    (tir-fns/IterVar domain v (iteration-variable-types iteration-type) thread-tag)))


(defn name->thread-axis-iterator
  "Create a thread iter-var from a thread axis name"
  [axis-name]
  (iteration-variable nil axis-name :thread-index :thread-tag axis-name))


(defmacro tvm-fn
  "Like (fn) but retains the arglists.  Lambda in clojure unfortunately does not."
  [arg-vec & body]
  (let [retval `(fn ~arg-vec
                  ~@body)]
    (with-meta retval {:arglists `(quote ~arg-vec)})))


(defn compute
  "Construct a new tensor by computing over the shape domain.

    The compute rule is result[axis] = fcompute(axis)

    Parameters
    ----------
    shape: Array of Expr
        The shape of the tensor

    fcompute: lambda function of indices-> value
        Specifies the input source expression

    name: str, optional
        The name hint of the tensor

    tag: str, optional
        Additonal tag information about the compute.

    attrs: dict, optional
        The additional auxiliary attributes about the compute.

    Returns
    -------
    The created compute node
    "
  [shape fcompute name & {:keys [tag attrs]
                     :or {tag ""}}]
  (let [fn-arglists (->> (meta fcompute)
                         :arglists
                         (mapv (comp safe-str clojure.core/name)))]
    (when-not-error fn-arglists
      (ex-info "Functions passed into compute must have the arglists in their metadata"
               {}))
    (when-not-error (= (count shape)
                       (count fn-arglists))
      (ex-info "fcompute must have same number of args as rank of shape"
               {:shape-rank (count shape)
                :num-fn-args (count fn-arglists)}))
    (let [compute-dim (map (fn [arg-name shape-value]
                             (iteration-variable [0 shape-value] arg-name
                                                 :data-parallel))
                           fn-arglists shape)
          body-data (apply fcompute (map :var compute-dim))
          body-data (if-not (instance? clojure.lang.Sequential body-data)
                      [body-data]
                      body-data)]
      (te-fns/ComputeOp (safe-str name) tag attrs compute-dim body-data))))


(defn commutative-reduce
  "1 left hand side, first var of reduce operation
  N right hand sides, rest of the variables of the reduce operation
  identity-val - initialization of left hand side.n
  expr-ary - one for each (const) right hand side.
  dtype - datatype of all inputs to reduction"
  [reduce-op identity-val dtype expr-seq axis-seq]
  (let [fn-arglists (->> (meta reduce-op)
                         :arglists
                         (map clojure.core/name)
                         (mapv #(variable % :dtype dtype)))
        reduce-ast [(apply reduce-op fn-arglists)]
        lhs-vars (take 1 fn-arglists)
        rhs-vars (drop 1 fn-arglists)
        comm-reducer (tir-fns/CommReducer
                      lhs-vars rhs-vars
                      (->node reduce-ast)
                      (->node [identity-val]))]
    (tir-fns/Reduce comm-reducer expr-seq
                    axis-seq nil 0 (->node []))))


(defn output-tensors
  [compute-op]
  (->> (clojure.core/range (te-fns/OpNumOutputs compute-op))
       (mapv (partial te-fns/OpGetOutput compute-op))))


(defn input-tensors
  [compute-op]
  (te-fns/OpInputTensors compute-op))


(defn ->operation
  [tens-or-op]
  (if (= (bindings/node-type-name tens-or-op) "Tensor")
    (:op tens-or-op)
    tens-or-op))
