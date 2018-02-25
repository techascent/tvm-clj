(ns tvm-clj.api
  (:require [tvm-clj.core :as c]
            [tvm-clj.base :as b]
            [think.resource.core :as resource]
            [clojure.set :as c-set])
  (:import [tvm_clj.core NodeHandle]
           [tvm_clj.tvm runtime runtime$TVMModuleHandle]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defmacro when-not-error
  [condition throw-clause]
  `(when-not (do ~condition)
     (throw ~throw-clause)))


(defn ->node
  [item]
  (b/->node item))


(defn variable
  "Create a scalar variable.  Returns a node handle"
  [^String name & {:keys [dtype]
                   :or {dtype "int32"}}]
  (c/global-node-function "_Var" name dtype))


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

  (let [domain (when domain
                 (if (= :range (c/get-node-type domain))
                   domain
                   (range (first domain) (second domain))))
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
  (let [indices (mapv (fn [index-val]
                        (let [node-data (->node index-val)]
                          (cond
                            (= :iteration-variable (c/get-node-type node-data))
                            (:var node-data)
                            (c/is-expression-node? node-data)
                            node-data
                            :else
                            (throw (ex-info "Index must be either iteration variable or expression"
                                            {:node-type (c/get-node-type node-data)})))))
                      indices)]
    (call (:dtype tensor) (get-in tensor [:op :name]) indices
          :halide (:op tensor) (:value_index tensor))))


(defn add
  [lhs rhs]
  (c/global-node-function "make.Add" lhs rhs))


(defmacro tvm-fn
  "Like (fn) but retains the arglists.  Lambda in clojure unfortunately does not."
  [arg-vec & body]
  (let [retval `(fn ~arg-vec
                  ~@body)]
    (with-meta retval {:arglists `(quote ~arg-vec)})))


(defn compute
  [shape fcompute & {:keys [name tag]
                     :or {name "compute"
                          tag ""}}]

  (let [fn-arglists (->> (meta fcompute)
                         :arglists
                         (mapv clojure.core/name))]
    (when-not-error fn-arglists
      (ex-info "Functions passed into compute must have the arglists in their metadata"
               {}))
    (when-not-error (= (count shape)
                       (count fn-arglists))
      (ex-info "fcompute must have same number of args as rank of shape"
               {:shape-rank (count shape)
                :num-fn-args (count fn-arglists)}))
    (let [compute-dim (map (fn [arg-name shape-value]
                             (iteration-variable [0 shape-value] arg-name :data-parallel))
                           fn-arglists shape)
          body-data (apply fcompute (map :var compute-dim))
          body-data (if-not (instance? clojure.lang.Sequential body-data)
                      [body-data]
                      body-data)]
      (-> (c/g-fn "_ComputeOp" name tag compute-dim body-data)
          (c/unpack-node-fields :recurse false)))))


(defn output-tensors
  [compute-op]
  (->> (clojure.core/range (c/global-function "_OpNumOutputs" compute-op))
       (mapv #(c/global-node-function "_OpGetOutput" compute-op (int %1)))))


(defn input-tensors
  [compute-op]
  (->> (c/global-node-function "_OpInputTensors" compute-op)
       c/tvm-array->jvm
       (mapv c/unpack-node-field)))


(defn create-schedule
  [op-seq]
  (let [op-seq (if-not (sequential? op-seq)
                 [op-seq]
                 op-seq)]
    (c/unpack-node-fields (c/g-fn "_CreateSchedule" op-seq)
                          :recurse false)))


(defn split-stage-by-factor
  [stage iter-var factor]
  (c/tvm-array->jvm (c/g-fn "_StageSplitByFactor" stage iter-var factor)))


(defn stage-bind
  "Bind an iter-var to a stage variable"
  [stage iter-var thread-ivar]
  (c/g-fn "_StageBind" stage iter-var thread-ivar))


(defn name->thread-axis-iterator
  "Create a thread iter-var from a thread axis name"
  [axis-name]
  (iteration-variable nil axis-name :thread-index :thread-tag axis-name))



(def default-build-config
  "Comments from tvm/build_module.h"
  {;;/*! \brief Threshold of number of steps in the loop to be automatically unrolled */
   :auto-unroll-max-step 0
   ;;/*! \brief The maximum nested level of loops that can be automatically unrolled */
   :auto-unroll-max-depth 8
   ;;/*! \brief The maximum extent of loop that will be unrolled */
   :auto-unroll-max-extent 0
   ;; /*!
   ;; * \brief Whether to explicitly unroll the loop. If set to false, the unroll hint will
   ;; * be passed to the CodeGen phase. Set to true if CodeGen supports unroll pragma.
   ;; */
   :unroll-explicit? true
   ;;/*! \brief Whether to detect global barrier */
   :detect-global-barrier? false
   ;;/*! \brief Whether to partition const loop */
   :partition-const-loop? false
   ;; /*!
   ;; * \brief The offset factor to use when constructing buffers. If this is set to
   ;; * 0, then the offset field is not used.
   ;; */
   :offset-factor 0
   ;; /*!
   ;; * \brief The data alignment to use when constructing buffers. If this is set to
   ;; * -1, then TVM's internal default will be used
   ;; */
   :data-alignment -1
   ;;/*! \brief Set to true if buffer arguments do not overlap. This enables more optimization. */
   :restricted-func? true
   ;; /*!
   ;; * \brief Splitting factor for loop splitting. If this is set to zero, no splitting will be
   ;; * done. Otherwise, a split will be done with this factor and the inner loop will be unrolled.
   ;; */
   :double-buffer-split-loop 1})


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

    --CN - REMOVED - No one understands what this does.  It is only referenced in the code
in order to perform a check during argument binding.  So the description below is accurate
for what it is worth but it is hard to me to see how this is useful.


    offset_factor: int, optional
        The factor of elem_offset field, when set,
        elem_offset is required to be multiple of offset_factor.
        If 0 is pssed, the alignment will be set to 1.
        if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.

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
            :or {name "buffer" dtype "float32" strides [] scope "" data-alignment -1
                 elem-offset 0}}]
  (let [shape (if (sequential? shape)
                shape
                [shape])
        data (if data data (variable name :dtype "handle"))
        offset-factor 0]
    (c/global-node-function "_Buffer"
                            data dtype shape strides elem-offset name scope
                            data-alignment offset-factor)))


(defn bind-arguments
  "Given an arg-list and existing bind map, produce a new arg list
and bind map with all arguments bound to input buffers with defined buffer layout.
Bind map is a map of type NodeHandle->NodeHandle where the keys are tensors and the
values are buffers"
  [arg-list bind-map build-config]
  (reduce (fn [[arg-list bind-map] arg]
            (condp = (c/get-node-type arg)
              :tensor
              (if-let [buf (bind-map arg)]
                [(conj arg-list buf) bind-map]
                (let [new-buf (declare-buffer (:shape arg) :dtype (:dtype arg)
                                              name (:name arg)
                                              :data-alignment (:data-alignment build-config))]
                  [(conj arg-list new-buf) (assoc bind-map arg new-buf)]))
              :buffer
              [(conj arg-list arg) bind-map]
              :variable
              [(conj arg-list arg) bind-map]))
          [[] bind-map]
          arg-list))


(defn- gfnr
  "Like global-node-function but the first argument is assumed to be the 'this' object
and the second is the function to call.  We need this slight transposition in order to use
the threading macro with the long set of ir pass possibilities."
  [item fn-name & args]
  ;;These are all nodes but don't upack fields; this causes too much unnecessary unpacking.
  (apply c/g-fn fn-name item args))


(def lowered-function-type->int-map
  {:mixed-function 0
   :host-function 1
   :device-functions 2})


(def int->lowered-function-type-map (c-set/map-invert lowered-function-type->int-map))


(defn schedule->lowered-function
  "Lowering step before build into target.

    Parameters
    ----------
    schedule : tvm.Schedule
        The schedule to be builded

    args : list of Buffer or Tensor or Var
        The argument lists to the function.

    name : str, optional
        The name of result function.

    bind-map: map of {:tensor :buffer}, optional
        mapping fuction or hash-map that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument list.

    simple_mode : bool, optional (not currently implemented)
        Whether only output simple and compact statement, this will skip
        LoopPartition, api wrapper generation and Unrolling.

    Returns
    -------
    f : LoweredFunc or Stmt
       The result function, if with_api_wrapper=False
       Then the Stmt before make api is returned.
    "
  ^NodeHandle [schedule args build-config & {:keys [name bind-map]
                                             :or {name "default_function"
                                                  bind-map {}}}]
  (let [schedule (c/g-fn "_ScheduleNormalize" schedule)
        [arg-list bind-map] (bind-arguments args bind-map build-config)
        bounds (c/g-fn "schedule.InferBound" schedule)
        cache-line-size 64]
    (-> schedule
        ;;Phase 0
        (gfnr "schedule.ScheduleOps" bounds)
        (gfnr "ir_pass.InjectPrefetch")
        ;;Phase 1
        (gfnr "ir_pass.StorageFlatten" bind-map cache-line-size)
        (gfnr "ir_pass.CanonicalSimplify")
        ;;Phase 2
        (gfnr "ir_pass.LoopPartition" (:partition-const-loop? build-config))
        (gfnr "ir_pass.VectorizeLoop")
        (gfnr "ir_pass.InjectVirtualThread")
        (gfnr "ir_pass.InjectDoubleBuffer" (:double-buffer-split-loop build-config))
        (gfnr "ir_pass.StorageRewrite")
        (gfnr "ir_pass.UnrollLoop"
              (:auto-unroll-max-step build-config)
              (:auto-unroll-max-depth build-config)
              (:auto-unroll-max-extent build-config)
              (:unroll-explicit? build-config))
        ;;Phase 3
        (gfnr "ir_pass.Simplify")
        (gfnr "ir_pass.LowerStorageAccessInfo")
        (gfnr "ir_pass.RemoveNoOp")
        (gfnr "ir_pass.RewriteUnsafeSelect")
        ;;Exit
        (gfnr "ir_pass.MakeAPI" name arg-list 0 (:restricted-func? build-config))
        (c/unpack-node-fields :recurse false)
        (update :func_type int->lowered-function-type-map))))


(def target-name->props
  [[#{:llvm} {:keys #{:cpu}}]
   [#{:cuda :nvptx} (fn [target-name]
                        {:keys #{:cuda :gpu}
                         :max-num-threads 512
                         :thread-warp-size 32})]
   [#{:rocm :opencl} (fn [target-name]
                         {:keys #{:rocm :gpu}
                          :max-num-threads 256})]
   [#{:metal :vulkan} (fn [target-name]
                          {:keys #{:gpu target-name}
                           :max-num-threads 256})]
   [#{:opengl} (fn [target-name]
                  {:keys #{:opengl}})]])


(defn create-target
  [target-name]
  (let [target-map-fn (->> target-name->props
                           (filter #((first %) target-name))
                           first
                           second)]
    (when-not-error target-map-fn
      (ex-info "Failed to find target properties in target"
               {:target-name target-name}))
    (merge {:target-name target-name
            :thread-warp-size 1}
           (target-map-fn target-name))))


(defn lowered-functions->module
  ^runtime$TVMModuleHandle
  [lowered-function-seq build-config & {:keys [target-name target-host]
                                        :or {target-name :llvm
                                             target-host :llvm}}]
  (let [arg-type-list (map c/get-node-type lowered-function-seq)]
    (when-not-error (= #{:lowered-function} (set arg-type-list))
      (ex-info "Argumentis not a sequence of lowered functions"
               {:arg-types arg-type-list})))
  (let [arg-name-set (->> (map :name lowered-function-seq)
                          set)
        target (create-target target-name)
        _ (when-not-error (= (count lowered-function-seq)
                             (count arg-name-set))
            (ex-info "Arguments have duplicate names or are themselves duplicated"
                     {:arg-names (mapv :name lowered-function-seq)}))
        [host-fns device-fns] (reduce (fn [[host-fns device-fns] lowered-fn]
                                        (condp = (:func_type lowered-fn)
                                          :host-function
                                          [(conj host-fns lowered-fn) device-fns]
                                          :device-function
                                          [host-fns (conj device-fns lowered-fn)]
                                          :mixed-function
                                          (let [warp-size (long (:thread-warp-size target))
                                                fsplits (-> (if (:detect-global-barrier? build-config)
                                                              (c/g-fn "ir_pass.ThreadSync" lowered-fn "global")
                                                              lowered-fn)
                                                            (gfnr "ir_pass.LowerThreadAllreduce" warp-size)
                                                            (gfnr "ir_pass.SplitHostDevice")
                                                            (c/tvm-array->jvm))]
                                            [(conj host-fns (first fsplits))
                                             (concat device-fns (rest fsplits))])))
                                      [[] []]
                                      lowered-function-seq)
        device-type (c/device-type->device-type-int target-name)
        host-fns (mapv (fn [host-fn]
                         (c/g-fn "ir_pass.BindDeviceType" host-fn device-type)
                         (c/g-fn "ir_pass.LowerTVMBuiltin" host-fn))
                       host-fns)
        target-host (if-not target-host
                      (if (= device-type runtime/kDLCPU)
                        target
                        ;;More checking here to work with stackvm
                        :llvm)
                      target-host)
        target-device target
        device-fns (mapv #(c/g-fn "ir_pass.LowerIntrin" % (name target-device)) device-fns)
        host-fns (->> host-fns
                      (map #(c/g-fn "ir_pass.LowerIntrin" % (name target-host)))
                      (map #(c/g-fn "ir_pass.CombineContextCall" %))
                      ;;Force here to avoid laziness + side-effectiness mixing
                      doall)
        ^runtime$TVMModuleHandle mhost (c/g-fn "codegen._Build" host-fns (name target-host))]
    (when (seq device-fns)
      (resource/with-resource-context
        (->> (c/g-fn "codegen._Build" device-fns (name target-device))
             (runtime/TVMModImport mhost))))
    mhost))


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
  clojure.lang.Sequential
  (->node [item] (apply c/tvm-array (map ->node item))))
