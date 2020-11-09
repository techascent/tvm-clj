(ns tvm-clj.api
  "Higher level API to build and compile tvm functions."
  (:require [tvm-clj.tvm-jna :refer [->node] :as bindings]
            [tvm-clj.jna.node :as jna-node]
            [tvm-clj.jna.fns.te :as te-fns]
            [tvm-clj.jna.fns.tir :as tir-fns]
            [tvm-clj.jna.fns.ir :as ir-fns]
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


(defn range
  "Create a range with defined start inclusive and end exclusive"
  [start end]
  (ir-fns/Range start end))


(defn const
  "Convert an item to a const (immediate) value"
  [numeric-value & [dtype]]
  (jna-node/const numeric-value dtype))


(defn static-cast
  "Cast an item from one datatype to another"
  [dtype expr-node]
  (bindings/global-node-function "make.Cast" (->dtype dtype) (->node expr-node)))


(defn cast
  "See static cast redone to allow usage in ->"
  [expr-node dtype]
  (static-cast dtype expr-node))


(def call-types
  "Possible call types from Halide/IR.h"
  {:extern 0 ;;< A call to an external C-ABI function, possibly with side-effects
   :extern-c-plus-plus 1 ;;< A call to an external C-ABI function, possibly with side-effects
   :pure-extern 2 ;;< A call to a guaranteed-side-effect-free external function
   :halide 3 ;;< A call to a Func
   :intrinsic 4  ;;< A possibly-side-effecty compiler intrinsic, which has special handling during codegen
   :pure-intrinsic 5 ;;< A side-effect-free version of the above.
   })

(defn ->call-type
  ^long [ctype]
  (cond
    (keyword? ctype)
    (if-let [retval (get call-types ctype)]
      retval
      (throw (ex-info "Failed to find call type"
                      {:call-type ctype})))
    (number? ctype)
    (long ctype)))


(def call-type-set (set (keys call-types)))


(defn call
  "Call a 'function', which is basically executing a statement.  For instance, getting a
  value from the tensor is calling a halide function with the tensor's generating-op and
  value index."
  [ret-dtype fn-name fn-args call-type function-ref value-index]
  (bindings/global-node-function "make.Call" (->dtype ret-dtype) fn-name fn-args
                          (->call-type call-type)
                          function-ref value-index))


(defn call-pure-intrin
  "Build expression by calling a pure intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : Expr
        The call expression.
    "
  [dtype func-name & args]
  (call dtype func-name (->node args) :pure-intrinsic nil 0))


(defn call-intrin
  "Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : Expr
        The call expression.
    "
  [dtype func-name & args]
  (call dtype func-name (->node args) :intrinsic nil 0))


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


(defmethod jna-node/get-extended-node-value :tensor
  [node-handle item-key]
  (cond
    (or (number? item-key)
        (sequential? item-key))
    (tget node-handle item-key)
    (= item-key :axis) (:axis (:op node-handle))
    :else
    nil))


(defmacro def-bin-op
  "Define a binary operation"
  [op-name op-fn]
  `(defn ~op-name
     [~'lhs ~'rhs]
     (~op-fn ~'lhs ~'rhs)))


(defmacro def-op
  "Define a binary operation"
  [op-name make-name]
  (let [lhs (symbol "lhs")]
    `(defn ~op-name
       [~lhs]
       (bindings/global-node-function ~make-name ~lhs))))


(defmacro def-bin-intrin-op
  [op-name]
  (let [lhs (symbol "lhs")
        rhs (symbol "rhs")]
    `(defn ~op-name
       [~lhs ~rhs]
       (call-pure-intrin (dtype/get-datatype ~lhs)
                         ~(str op-name)
                         ~lhs
                         ~rhs))))

(defmacro def-intrin-op
  [op-name]
  (let [lhs (symbol "lhs")]
    `(defn ~op-name
       [~lhs]
       (call-pure-intrin (dtype/get-datatype ~lhs)
                         ~(str op-name)
                         ~lhs))))



(def-bin-op add tir-fns/Add)
;; (def-bin-op sub "make.Sub")
;; (def-bin-op mod "make.Mod")
;; (def-bin-op mul "make.Mul")
;; (def-bin-op div "make.Div")
;; (def-bin-op eq "make.EQ")
;; (def-bin-op min "make.Min")
;; (def-bin-op max "make.Max")
;; (def-intrin-op exp)
;; (def-intrin-op tanh)
;; (def-intrin-op sigmoid)
;; (def-intrin-op log)
;; (def-intrin-op sqrt)
;; (def-intrin-op floor)
;; (def-intrin-op ceil)
;; (def-intrin-op trunc)
;; (def-op abs "make.abs")
;; (def-intrin-op round)
;; (def-bin-intrin-op power)
;; (def-intrin-op popcount)



(defn select
  "Select between two expressions based on a condition.  Thus works similar to the
clojure 'if' statement."
  [bool-stmt true-stmt false-stmt]
  (bindings/global-node-function "make.Select" bool-stmt true-stmt false-stmt))


(defn- get-for-type-idx
  ^long [for-type]
  (case for-type
    :serial 0
    :parallel 1
    :vectorize 2
    :unroll 3))


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
        comm-reducer (bindings/g-fn "make.CommReducer"
                             lhs-vars rhs-vars
                             (->node reduce-ast)
                             (->node [identity-val]))]
    (bindings/g-fn "make.Reduce" comm-reducer expr-seq axis-seq (->node true) 0)))


(defn output-tensors
  [compute-op]
  (->> (clojure.core/range (te-fns/OpNumOutputs compute-op))
       (mapv (partial te-fns/OpGetOutput compute-op))))


(defn input-tensors
  [compute-op]
  (te-fns/OpInputTensors compute-op))

(defn throw-nil
  [item key-val]
  (if-let [retval (get item key-val)]
    retval
    (throw (ex-info "Expected object but got nil"
                    {:item item
                     :key key-val}))))


(defn ->operation
  [tens-or-op]
  (if (= (bindings/node-type-name tens-or-op) "Tensor")
    (throw-nil tens-or-op :op)
    tens-or-op))


(defn create-schedule
  [op-seq]
  (let [op-seq (->> (if-not (sequential? op-seq)
                      [op-seq]
                      op-seq)
                    (mapv ->operation))]
    (te-fns/CreateSchedule op-seq)))


(defn ->stage
  [stage-or-schedule operation]
  (case (bindings/node-type-name stage-or-schedule)
    "Stage" stage-or-schedule
    "Schedule" (throw-nil (:stage_map stage-or-schedule)
                          (->operation operation))))


(defmethod jna-node/get-extended-node-value :schedule
  [node-handle item-key]
  (->stage node-handle (->operation item-key)))


(defn stage-split-axis
  [stage iter-var factor]
  (te-fns/StageSplitByFactor stage iter-var factor))


(defn stage-bind
  "Bind an iter-var to a stage variable"
  [stage iter-var thread-ivar]
  (te-fns/StageBind stage iter-var thread-ivar))


(defn stage-compute-at
  "Compute src stage at dst stage dst axis"
  [src-stage dst-stage dst-axis]
  (te-fns/StageComputeAt src-stage dst-stage dst-axis))


(defn stage-fuse
  "Fuse n-axis together, returns single new axis"
  [stage axis-args]
  ;;If there is only one axis, then fusing is pointless
  (if (= 1 (count axis-args))
    (first axis-args)
    (te-fns/StageFuse stage axis-args)))


(defn stage-parallel
  "Indicate that this axis has complete parallelism"
  [stage axis]
  (te-fns/StageParallel stage axis))


(defn stage-inline
  [stage]
  (te-fns/StageComputeInline stage))


(defn stage-tile
  [stage outer-axis inner-axis outer-dim inner-dim]
  (te-fns/StageTile stage outer-axis inner-axis outer-dim inner-dim))


(defn stage-reorder
  [stage axis-seq]
  (te-fns/StageReorder stage axis-seq))


(defn stage-vectorize
  [stage axis]
  (te-fns/StageVectorize stage axis))


(defn stage-unroll
  [stage axis]
  (te-fns/StageUnroll stage axis))


(defn schedule-cache-write
  "Returns a new tensor"
  [schedule tensor cache-type]
  (let [retval (te-fns/ScheduleCacheWrite schedule tensor cache-type)]
    {:tensor retval
     :schedule schedule}))


(defn schedule-cache-read
  [schedule tensor cache-type readers]
  (throw (ex-info "Unimplemented" {})))


(defn stage-bind-gpu
  "Bind the gpu-defined axis to the tvm axis.
  GPU (cuda, opencl) define a roughly level stage breakdown of axis: block and thread.
  Threads run on the same block and can share a special kind of memory (called shared
  memory).  There can be up to 3 tvm axis per block or thread and these are labeled
  (outer iterator to inner iterator):
  [z y x]"
  [stage block-axis-seq thread-axis-seq]
  (let [axis-names ["z" "y" "x"]
        full-info-fn (fn [grp-name axis-seq]
                         (map vector
                              (repeat grp-name)
                              axis-seq
                              ;;map to axis such that if you have one, it becomes
                              ;;the x axis.  If you have 2, first is y and second
                              ;;is x, etc.
                              (drop (- 3 (count axis-seq)) axis-names)))]
    (when-not (and (<= (count block-axis-seq) 3)
                   (<= (count thread-axis-seq) 3))
      (throw (ex-info "Block, threads can have up to 3 axis"
                      {:thread-axis-count (count thread-axis-seq)
                       :block-axis-count (count block-axis-seq)})))
    (->> (concat (full-info-fn "blockIdx" block-axis-seq)
                 (full-info-fn "threadIdx" thread-axis-seq))
         (map (fn [[grp-name axis gpu-axis-name]]
                (stage-bind stage axis
                            (name->thread-axis-iterator
                             (str grp-name "." gpu-axis-name)))))
         dorun)))


(defn stage-gpu-injective
  [stage op & {:keys [thread-count axis]
               :or {thread-count 16}}]

  (let [retval stage
        op (->operation op)
        stage (->stage stage op)
        fused-axis (stage-fuse stage (or axis (:axis op)))
        [bx tx] (stage-split-axis stage fused-axis thread-count)]
    (stage-bind-gpu stage [bx] [tx])
    retval))


(defn stage-cpu-injective
  [stage op & {:keys [axis]}]
  (let [retval stage
        op (->operation op)
        stage (->stage stage op)
        fused-axis (stage-fuse stage (or axis (:axis op)))]
    (stage-parallel stage fused-axis)
    retval))
