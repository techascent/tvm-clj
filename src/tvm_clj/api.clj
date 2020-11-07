(ns tvm-clj.api
  "Higher level API to build and compile tvm functions."
  (:require [tvm-clj.tvm-jna :refer [->node] :as bindings]
            [tvm-clj.bindings.definitions :as bindings-defs]
            [tvm-clj.jna.node :as jna-node]
            [tvm-clj.jna.fns.te :as te-fns]
            [tvm-clj.jna.fns.tir :as tir-fns]
            [tvm-clj.jna.fns.ir :as ir-fns]
            [tvm-clj.jna.fns.node :as node-fns]
            [tvm-clj.jna.fns.schedule :as schedule-fns]
            [tvm-clj.jna.fns.tir.transform :as tir-transform-fns]
            [tvm-clj.jna.fns.transform :as transform-fns]
            [tvm-clj.jna.fns.codegen :as codegen-fns]
            [tvm-clj.jna.fns.schedule :as schedule-fns]
            [tvm-clj.jna.fns.target :as target-fns]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.resource :as resource]
            [clojure.set :as c-set]
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


;;Below taken from
;;https://github.com/apache/incubator-tvm/blob/728b829575e5e690870b111ae2256cbe0f3dbe6f/python/tvm/driver/build_module.py

(def lowered-function-type->int-map
  {:mixed-function 0
   :host-function 1
   :device-functions 2})


(def int->lowered-function-type-map (c-set/map-invert lowered-function-type->int-map))


(defn declare-buffer
  "Decleare a new symbolic buffer.

    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.

    See the note below for detailed discussion on usage of buffer.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    data : Var, optional
        The data pointer in the buffer.

    strides: array of Expr
        The stride of the buffer.

    elem_offset: Expr, optional
        The beginning offset of the array to data.
        In terms of number of elements of dtype.

    scope: str, optional
        The storage scope of the buffer, if not global.
        If scope equals empty string, it means it is global memory.

    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.

    --CN - REMOVED - No one understands what this does.  It is only referenced in the
  code in order to perform a check during argument binding.  So the description below is
  accurate for what it is worth but it is hard to me to see how this is useful.


    offset_factor: int, optional
        The factor of elem_offset field, when set,
        elem_offset is required to be multiple of offset_factor.
        If 0 is pssed, the alignment will be set to 1.
        if non-zero is passed, we will created a Var for elem_offset if elem_offset is
  not None.

     --CN - END-REMOVED --

    Returns
    -------
    buffer : Buffer
        The created buffer

    Note
    ----
    Buffer data structure reflects the DLTensor structure in dlpack.
    While DLTensor data structure is very general, it is usually helpful
    to create function that only handles specific case of data structure
    and make compiled function benefit from it.

    If user pass strides and elem_offset is passed as None
    when constructing the function, then the function will be specialized
    for the DLTensor that is compact and aligned.
    If user pass a fully generic symbolic array to the strides,
    then the resulting function becomes fully generic."
  [shape & {:keys [dtype name data strides elem-offset scope data-alignment]
            :or {name "buffer" dtype "float32" scope "" data-alignment -1}}]
  (let [shape (if (instance? java.util.RandomAccess shape)
                shape
                [shape])
        elem-offset (if elem-offset elem-offset 0)
        data (if data data
                 (tir-fns/Var name (ir-fns/PointerType
                                    (ir-fns/PrimType dtype))))
        offset-factor 0]
    (tir-fns/Buffer
     data (->dtype dtype) shape strides elem-offset
     (safe-str name) scope
     data-alignment offset-factor
     "")))


(defn bind-arguments
  "Given an arg-list and existing bind map, produce a new arg list
and bind map with all arguments bound to input buffers with defined buffer layout.
Bind map is a map of type NodeHandle->NodeHandle where the keys are tensors and the
values are buffers.  The default is to bind a compact, non-offset buffer so if you want
a different buffer type than this then you need to bind it yourself."
  [arg-list compact? bind-map]
  (reduce (fn [[arg-list bind-map] arg]
            (condp = (bindings/node-type-name arg)
              "Tensor"
              (if-let [buf (bind-map arg)]
                [(conj arg-list buf) bind-map]
                (let [shape (:shape arg)
                      new-buf (declare-buffer
                               (:shape arg) :dtype (:dtype arg))]
                  [(conj arg-list new-buf) (assoc bind-map arg new-buf)]))
              "Buffer"
              [(conj arg-list arg) bind-map]
              "tir.Var"
              [(conj arg-list arg) bind-map]))
          [[] (or bind-map {})]
          arg-list))

(defn current-pass-context-config
  []
  (:config (transform-fns/GetCurrentPassContext)))


(defn schedule->function
  "According to the given schedule, form a function.

    Parameters
    ----------
    sch : tvm.te.schedule.Schedule
        The given scheduler to form the raw body

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    name : str
        The name of result function.

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        The binds information

    Returns
    -------
    The body formed according to the given schedule"
  [sch args name binds]
  ;; normalize schedule first
  (let [config (current-pass-context-config)
        sch (te-fns/ScheduleNormalize sch)
        bounds (schedule-fns/InferBound sch)
        stmt (schedule-fns/ScheduleOps sch bounds)
        compact? (schedule-fns/VerifyCompactBuffer stmt)
        [arg-list binds] (bind-arguments args compact? binds)

        stmt (schedule-fns/SchedulePostProcRewriteForTensorCore stmt sch binds)
        func (schedule-fns/SchedulePostProcToPrimFunc arg-list stmt binds)
        func (ir-fns/BaseFuncWithAttr func
                                      (->node "global_symbol")
                                      (->node name))

        func (if (get config "tir.noalias" true)
               (ir-fns/BaseFuncWithAttr func
                                        (->node "tir.noalias")
                                        (->node true))
               func)]
    (ir-fns/IRModule {(ir-fns/GlobalVar (safe-str name)) func}
                     {})))


(defn sequential-pass
  ([mod {:keys [optimization-level]
          :or {optimization-level 2}}
    pass-list]
   (-> (transform-fns/Sequential pass-list optimization-level "sequential" nil)
       (transform-fns/RunPass mod))
   ([mod pass-list]
    (sequential-pass mod nil pass-list))))



(defn lower
  "Lowering step before build into target.

    Parameters
    ----------
    sch : tvm.te.schedule.Schedule
        The schedule to be built

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    name : str, optional
        The name of result function.

    binds : dict of :any:`Tensor` to :any:`Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    simple_mode : bool, optional
        Whether only output simple and compact statement, this will skip
        LoopPartition, api wrapper generation and Unrolling.

    Returns
    -------
    m : IRModule or Stmt
       The result IRModule, if simple_mode=False
       Then the Stmt before make api is returned."
  [sch args {:keys [name binds simple-mode? optimization-level]
             :or {name "main"
                  optimization-level 2}}]
  ;; config setup
  (let [config (current-pass-context-config)
        instrument-bound-checkers? (boolean (get config "tir.instrument_bound_checkers"))
        disable-vectorize? (boolean (get config "tir.disable_vectorize"))
        ;;Lower passes are tuples [pass-idx pass-fn]
        lower-phases (->> (get config "tir.add_lower_pass")
                          (group-by first)
                          (map (fn [[k v]] [k (mapv second v)]))
                          (into {}))

        ;; Phase 0
        mod (if (= "Schedule" (bindings/node-type-name sch))
              (schedule->function sch args name binds)
              sch)]
    (->> (concat (get lower-phases 0)
                 [(tir-transform-fns/InjectPrefetch)
                  (tir-transform-fns/StorageFlatten 64 instrument-bound-checkers?)
                  (tir-transform-fns/BF16Legalize)
                  (tir-transform-fns/NarrowDataType 32)
                  (tir-transform-fns/Simplify)]
                 (get lower-phases 1)
                 (when-not simple-mode?
                   [(tir-transform-fns/LoopPartition)])
                 [(tir-transform-fns/VectorizeLoop (not disable-vectorize?))
                  (tir-transform-fns/InjectVirtualThread)
                  (tir-transform-fns/InjectDoubleBuffer)
                  (tir-transform-fns/StorageRewrite)
                  (tir-transform-fns/UnrollLoop)]
                 (get lower-phases 2)
                 [(tir-transform-fns/Simplify)
                  (tir-transform-fns/RemoveNoOp)
                  (tir-transform-fns/RewriteUnsafeSelect)
                  (tir-transform-fns/HoistIfThenElse)]
                 (get lower-phases 3)
                 (when instrument-bound-checkers?
                   [(tir-transform-fns/InstrumentBoundCheckers)]))
     (sequential-pass mod {:optimization-level optimization-level}))))


(defn make-fn-pass
  "Define a TVM compiler pass given a clojure fn.  Fn is a function from IRModule->IRModule.

  Options:

  * `opt_level` : `int`
        The optimization level of this module pass.  Defaults to zero which guarantees
        it will run.  Optimization levels go from 0 to at least 2.

  * `name` : Optional[str]
        The name of the function pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

  * `required` : Optional[List[str]]
        The list of passes that the function pass is dependent on."
  [map-fn {:keys [opt-level fname required]
           :or {opt-level 0
                fname "Apply"
                required []}}]
  (tir-transform-fns/CreatePrimFuncPass map-fn (transform-fns/PassInfo opt-level fname required)))


(defn map-pass
  "Map a function across an IRModule potentially modifying the input
  IRModule.  The IRModule is returned.

  map-fn is expected to be a function from IRModule->IRModule.

  Options : See options for `fn-pass`"
  ([map-fn options ir-module]
   (-> (make-fn-pass map-fn options)
       (transform-fns/RunPass ir-module)))
  ([map-fn ir-module]
   (map-pass map-fn nil ir-module)))


(defn rvalue-reference
  "Create an RValue reference to the object and mark the object as moved.

  This marks the object letting the TVM system know that it will not be
  accessed by caller after this point.

  A unique reference may trigger a copy on write optimization that avoids
  a copy when we mutably transform an object.

  Note
  ----
  All the reference of the object becomes invalid after it is moved.
  Be very careful when using this feature.

  Examples
  --------


```clojure
```"
  [node]
  (vary-meta node assoc :rvalue-reference? true))


(defn assoc-attr
  "Create a new copy of the function and update the attribute.

   Parameters
   ----------

   * attr_key_or_dict : Union[str, dict]
     The attribute key to use or a dict containing multiple key value pairs.

   * attr_value : Object
     The new attribute value.

   Returns
   -------
   * func : Function - A new copy of the function."
  [relay-expr attr-name attr-value & args]
  (let [args (concat [attr-name attr-value]
                     args)
        _ (errors/when-not-errorf
           (== 0 (rem (count args) 2))
           "Assoc takes an even number of att-name,att-value arguments: %s"
           (mapv str args))]
    (reduce (fn [relay-expr [attr-name attr-value]]
              (-> (rvalue-reference relay-expr)
                  (ir-fns/BaseFuncWithAttr attr-name (->node attr-value))))
            ;;Copy the relay-expr so we know we can guarantee it is an rvalue-reference
            ;;above.
            (ir-fns/BaseFuncCopy relay-expr)
            (map vec (partition 2 args)))))


(def ^long DEVICE_KERNEL_LAUNCH 2)
(def ^long C_PACKED_FUNC 1)
(def ^long DEFAULT 0)


(defn build_for_device
    "Build the lowered functions for a device with the given compilation
    target.

    Parameters
    ----------
    input_mod : IRModule
        The schedule to be built.

    target : str or :any:`tvm.target.Target`
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target`
        The host compilation target.

    Returns
    -------
    fhost : IRModule
        The host IRModule.

    mdev : tvm.module
        A module that contains device code."
  [input-mod target target-host]
  (let [target (target-fns/Target target)
        target_host (target-fns/Target target_host)
        device-type (bindings/device-type->int target)
        config (current-pass-context-config)
        mod-mixed input-modn
        ;;mark every function as being on the device
        mod-mixed (map-pass #(assoc-attr % "target" target) input-modn)

        mod-mixed (->> (concat
                        [(tir-transform-fns.VerifyMemory)]
                        (when (== 1 (count (:functions mod-mixed)))
                          [(make-fn-pass #(assoc-attr % "tir.is_entry_func" true))])
                        (when (get config "tir.detect_global_barrier")
                          [(tir-transform-fns/ThreadSync "global")])
                        [(tir-transform-fns/ThreadSync "shared")
                         (tir-transform-fns/ThreadSync "warp")
                         (tir-transform-fns/InferFragment)
                         (tir-transform-fns/LowerThreadAllreduce)
                         (tir-transform-fns/MakePackedAPI)
                         (tir-transform-fns/SplitHostDevice)])
                       (sequential-pass mod-mixed))

        ;; device optimizations
        opt-device = (->> [(make-filter-pass #(= (get-in % [:attrs "calling_conv"])
                                                 DEVICE_KERNEL_LAUNCH))
                           (tir-transform-fns/LowerWarpMemory)
                           (tir-transform-fns/Simplify)
                           (tir-transform-fns/LowerDeviceStorageAccessInfo)
                           (tir-transform-fns/LowerCustomDatatypes)
                           (tir-transform-fns/LowerIntrin)]
                          (sequential-pass mod-mixed))

        ;; host optimizations
        opt_host = (->> [(make-filter-pass #(not= (get-in % [:attrs "calling_conv"])
                                                  DEVICE_KERNEL_LAUNCH)),
                         (make-fn-pass #(assoc-attr % "target" target)),
                         (tir-transform-fns/LowerTVMBuiltin)
                         (tir-transform-fns/LowerDeviceStorageAccessInfo)
                         (tir-transform-fns/LowerCustomDatatypes)
                         (tir-transform-fns/LowerIntrin)
                         (tir-transform-fns/CombineContextCall)]
                        (sequential-pass mod-mixed))]



    if device_type == ndarray.cpu(0).device_type and target_host == target:
    assert len(mod_dev.functions) == 0
    if "gpu" in target.keys and len(mod_dev.functions) == 0:
    warnings.warn(
                  "Specified target %s, but cannot find device code, did you do " "bind?" % target
                  )

    rt_mod_dev = codegen.build_module(mod_dev, target) if len(mod_dev.functions) != 0 else None
    return mod_host, rt_mod_dev))



(defn build
    "Build a function with arguments as signature. Code will be generated
    for devices coupled with target information.

    Parameters
    ----------
    inputs : tvm.te.Schedule, IRModule, or dict of target to IRModule
        The schedule to be built

    args : list of Buffer or Tensor or Var, optional
        The argument lists to the function.

    target : str or :any:`tvm.target.Target`, optional
        The target and option of the compilation.

    target_host : str or :any:`tvm.target.Target` optional
        Host compilation target, if target is device.
        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        setup the dimensions and parameters correctly.
        target_host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    name : str, optional
        The name of result function.

    binds : dict, optional
        Dictionary that maps the binding of symbolic buffer to Tensor.
        By default, a new buffer is created for each tensor in the argument.

    Returns
    -------
    ret : tvm.module
        A module that combines both host and device code.

    Examples
    ________
    There are two typical example uses of this function depending on the type
    of the argument `inputs`:
    1. it is an IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s = tvm.te.create_schedule(C.op)
        m = tvm.lower(s, [A, B, C], name=\"test_add\")
        rt_mod = tvm.build(m, target=\"llvm\")

    2. it is a dict of compilation target to IRModule.

    .. code-block:: python

        n = 2
        A = te.placeholder((n,), name='A')
        B = te.placeholder((n,), name='B')
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
        s1 = tvm.te.create_schedule(C.op)
        with tvm.target.cuda() as cuda_tgt:
          s2 = topi.cuda.schedule_injective(cuda_tgt, [C])
          m1 = tvm.lower(s1, [A, B, C], name=\"test_add1\")
          m2 = tvm.lower(s2, [A, B, C], name=\"test_add2\")
          rt_mod = tvm.build({\"llvm\": m1, \"cuda\": m2}, target_host=\"llvm\")

    Note
    ----
    See the note on :any:`tvm.target` on target string format.
    "
  [fn-seq {:keys [target target-host]
           :or {target "llvm"
                target-host "llvm"}}]
  ;;map of target to IRModule
  (let [])
)
