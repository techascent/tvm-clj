(ns tvm-clj.api
  (:require [tvm-clj.core :as c]
            [tvm-clj.base :as b]
            [think.resource.core :as resource])
  (:import [tvm_clj.base NodeHandle]))


(defmacro when-not-error
  [condition throw-clause]
  `(when-not (do ~condition)
     (throw ~throw-clause)))


(defn ->node
  [item]
  (b/->node item))


(defn variable
  "Create a scalar variable.  Returns a node handle"
  [^String name & {:keys [type-str]
                   :or {type-str "int32"}}]
  (c/global-node-function "_Var" name type-str))


(defn placeholder
  "Create a user-supplied tensor variable"
  [shape & {:keys [dtype name]
            :or {dtype "float32"
                 name "placeholder"}}]
  (let [shape (if-not (instance? clojure.lang.Seqable shape)
                [shape]
                shape)]
    (c/global-node-function "_Placeholder" shape dtype name)))


(defn range
  "Create a range with defined start inclusive and end exclusive"
  [start end]
  (c/global-node-function "Range" start end))


(defn const
  "Convert an item to a const (immediate) value"
  [numeric-value & {:keys [dtype]
                    :or {dtype "float64"}}]
  (c/global-node-function "_const" numeric-value dtype))


(def iteration-variable-types
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
   ;; *  Note that this is already assumed to be paralellized.
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
   :opaque 4
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

  (let [domain (if (= :range (c/get-node-type domain))
                 domain
                 (range (first domain) (second domain)))
        v (variable name)]
    (c/global-node-function "_IterVar" domain v (iteration-variable-types iteration-type) thread-tag)))


(def call-types
  "Possible call types from Halide/IR.h"
  {:extern 0 ;;< A call to an external C-ABI function, possibly with side-effects
   :extern-c-plus-plus 1 ;;< A call to an external C-ABI function, possibly with side-effects
   :pure-extern 2 ;;< A call to a guaranteed-side-effect-free external function
   :halide 3 ;;< A call to a Func
   :intrinsic 4  ;;< A possibly-side-effecty compiler intrinsic, which has special handling during codegen
   :pure-intrinsic 5 ;;< A side-effect-free version of the above.
   })


(def call-type-set (set (keys call-types)))


(defn call
  "Call a 'function', which is basically executing a statement.  For instance, getting a value from the tensor
is calling a halide function with the tensor's generating-op and value index."
  [ret-dtype fn-name fn-args call-type function-ref value-index]
  (when-not-error (call-type-set call-type)
    (ex-info "Unrecognized call type"
             {:call-types call-type-set
              :call-type call-type}))
  (c/global-node-function "make.Call" ret-dtype fn-name fn-args
                          (call-types call-type) function-ref value-index))


(defn tget
  "Get an item from a tensor"
  [tensor indices]
  (when-not-error (= (count (:shape tensor))
                     (count indices))
    (ex-info "Num indices must match tensor rank"
             {:tensor-range (count (:shape tensor))
              :index-count (count indices)}))
  (resource/with-resource-context
    (let [indices (->node indices)]
      (call (:dtype tensor) (get-in tensor [:op :name]) indices
            :halide (:op tensor) (:value-index tensor)))))


(defn add
  [lhs rhs]
  (c/global-node-function "make.Add" lhs rhs))


(extend-protocol b/PConvertToNode
  NodeHandle
  (->node [item] item)
  Boolean
  (->node [item] (const item :dtype "uint1x1"))
  Byte
  (->node [item] (const item :dtype "int8"))
  Short
  (->node [item] (const item :dtype "int16"))
  Integer
  (->node [item] (const item :dtype "int32"))
  Long
  (->node [item] (const item :dtype "int64"))
  Float
  (->node [item] (const item :dtype "float32"))
  Double
  (->node [item] (const item :dtype "float64"))
  clojure.lang.Seqable
  (->node [item] (apply c/tvm-array (map ->node item))))
