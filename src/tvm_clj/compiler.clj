(ns tvm-clj.compiler
  (:require [tvm-clj.ast :as ast]
            [tvm-clj.schedule :as schedule]
            [tvm-clj.impl.base :as base]
            [tvm-clj.impl.protocols :as bindings]
            [tvm-clj.impl.fns.ir :as ir-fns]
            [tvm-clj.impl.fns.transform :as transform-fns]
            [tvm-clj.impl.fns.tir.transform :as tir-transform-fns]
            [tvm-clj.impl.fns.schedule :as schedule-fns]
            [tvm-clj.impl.fns.te :as te-fns]
            [tvm-clj.impl.fns.tir :as tir-fns]
            [tvm-clj.impl.fns.target :as target-fns]
            [tvm-clj.impl.module :as module]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.jna :as jna]
            [clojure.set :as set]))


(defn ^:no-doc sequential-pass
  "Perform a sequential pass, using the passes defined in the pass list
  and returning a new ir module."
  ([mod {:keys [optimization-level]
          :or {optimization-level 2}}
    pass-list]
   (-> (transform-fns/Sequential pass-list optimization-level "sequential" nil)
       (transform-fns/RunPass mod)))
  ([mod pass-list]
   (sequential-pass mod nil pass-list)))


(defn ^:no-doc make-fn-pass
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
  ([map-fn {:keys [opt-level fname required]
             :or {opt-level 0
                  fname "Apply"
                  required []}}]
   (tir-transform-fns/CreatePrimFuncPass (fn [func mod ctx]
                                           (map-fn (ir-fns/BaseFuncCopy func)))
                                         (transform-fns/PassInfo opt-level fname required)))
  ([map-fn]
   (make-fn-pass map-fn nil)))


(defn ^:no-doc make-filter-pass
  "Filter functions by the calling convention attribute.

    Parameters
    ----------
    fcond : tvm.tir.PrimFunc -> bool
        The condition of the filtering.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass"
  ([filter-fn {:keys [opt-level fname required]
                :or {opt-level 0
                     fname "Filter"
                     required []}}]
   (tir-transform-fns/CreatePrimFuncPass (fn [func mod ctx]
                                           ;;All the above arguments are passed by rvalue which
                                           ;;makes them potential landmines.
                                           (let [func (ir-fns/BaseFuncCopy func)]
                                             (when (filter-fn func)
                                               func)))
                                         (transform-fns/PassInfo opt-level fname required)))
  ([filter-fn]
   (make-filter-pass filter-fn nil)))


(defn ^:no-doc map-pass
  "Map a function across an IRModule potentially modifying the input
  IRModule.  The IRModule is returned.

  map-fn is expected to be a function from IRModule->IRModule.

  Options : See options for `fn-pass`"
  ([map-fn options ir-module]
   (-> (make-fn-pass map-fn options)
       (transform-fns/RunPass ir-module)))
  ([map-fn ir-module]
   (map-pass map-fn nil ir-module)))


(defn ^:no-doc rvalue-reference
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


(defn ^:no-doc assoc-attr
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
                     args)]
    (errors/when-not-errorf
     (== 0 (rem (count args) 2))
     "Assoc takes an even number of att-name,att-value arguments: %s"
     (mapv str args))
    relay-expr
    (reduce (fn [relay-expr [attr-name attr-value]]
              (ir-fns/BaseFuncWithAttr relay-expr
                                       (bindings/->node attr-name)
                                       (bindings/->node attr-value)))
            relay-expr
            (map vec (partition 2 args)))))


;;Below taken from
;;https://github.com/apache/incubator-tvm/blob/728b829575e5e690870b111ae2256cbe0f3dbe6f/python/tvm/driver/build_module.py

(def lowered-function-type->int-map
  {:mixed-function 0
   :host-function 1
   :device-functions 2})


(def int->lowered-function-type-map (set/map-invert lowered-function-type->int-map))


(defn ^:no-doc declare-buffer
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
     data (ast/->dtype dtype) shape strides elem-offset
     (ast/safe-str name) scope
     data-alignment offset-factor
     "")))


(defn ^:no-doc bind-arguments
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
                (let [new-buf (declare-buffer
                               (:shape arg)
                               :dtype (:dtype arg)
                               :name (try (get-in arg [:op :name])
                                          (catch Throwable e
                                            "Buffer")))]
                  [(conj arg-list new-buf) (assoc bind-map arg new-buf)]))
              "Buffer"
              [(conj arg-list arg) bind-map]
              "tir.Var"
              [(conj arg-list arg) bind-map]))
          [[] (or bind-map {})]
          arg-list))


(defn ^:no-doc current-pass-context-config
  []
  (:config (transform-fns/GetCurrentPassContext)))


(defn ^:no-doc schedule->function
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
                                      (bindings/->node "global_symbol")
                                      (bindings/->node name))

        func (if (get config "tir.noalias" true)
               (ir-fns/BaseFuncWithAttr func
                                        (bindings/->node "tir.noalias")
                                        (bindings/->node true))
               func)]
    (ir-fns/IRModule {(ir-fns/GlobalVar (ast/safe-str name)) func}
                     {})))


(defn ^:no-doc lower
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


(def ^{:tag 'long :private true} DEVICE_KERNEL_LAUNCH 2)
(def ^{:tag 'long :private true} C_PACKED_FUNC 1)
(def ^{:tag 'long :private true} DEFAULT 0)

(defn ^:no-doc tvm-fn-attrs
  [item]
  (-> (ir-fns/BaseFunc_Attrs item)
      (ir-fns/DictAttrsGetDict)))


(defn ^:no-doc build-for-device
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
  (let [target (target-fns/Target (ast/safe-str target))
        config (current-pass-context-config)
        ;;mark every function as being on the device
        mod-mixed (map-pass #(assoc-attr % "target" target) input-mod)

        mod-mixed (->> (concat
                        [(tir-transform-fns/VerifyMemory)]
                        (when (== 1 (count (:functions mod-mixed)))
                          [(make-fn-pass #(assoc-attr % "tir.is_entry_func" true))])
                        (when (get config "tir.detect_global_barrier")
                          [(tir-transform-fns/ThreadSync "global")])
                        [(tir-transform-fns/ThreadSync "shared")
                         (tir-transform-fns/ThreadSync "warp")
                         (tir-transform-fns/InferFragment)
                         (tir-transform-fns/LowerThreadAllreduce)
                         (tir-transform-fns/MakePackedAPI 0)
                         (tir-transform-fns/SplitHostDevice)])
                       (sequential-pass mod-mixed))

        ;; device optimizations
        device-modules (->> [(make-filter-pass #(= (get (tvm-fn-attrs %) "calling_conv")
                                                  DEVICE_KERNEL_LAUNCH))
                            (tir-transform-fns/LowerWarpMemory)
                            (tir-transform-fns/Simplify)
                            (tir-transform-fns/LowerDeviceStorageAccessInfo)
                            (tir-transform-fns/LowerCustomDatatypes)
                            (tir-transform-fns/LowerIntrin)]
                           (sequential-pass mod-mixed))

        ;; host optimizations
        host-modules (->> [(make-filter-pass #(not= (get (tvm-fn-attrs %) "calling_conv")
                                                   DEVICE_KERNEL_LAUNCH))
                          (make-fn-pass #(assoc-attr % "target" target))
                          (tir-transform-fns/LowerTVMBuiltin)
                          (tir-transform-fns/LowerDeviceStorageAccessInfo)
                          (tir-transform-fns/LowerCustomDatatypes)
                          (tir-transform-fns/LowerIntrin)
                          (tir-transform-fns/CombineContextCall)]
                          (sequential-pass mod-mixed))

        ;;Finalize device modules
        device-modules (when (not= 0 (count (:functions device-modules)))
                         (target-fns/Build device-modules target))]
    [host-modules device-modules]))



(base/make-tvm-jna-fn TVMModImport
                      "Import a tvm module into this"
                      Integer
                      [this jna/as-ptr]
                      [other jna/as-ptr])



(defn build
    "Build a map of function entries.
  fn-entries is a map of name to fn-data and fn-data is a
  map containing:


  * `:schedule` - schedule to use.
  * `:arguments` - argument declarations to the function
  * `:bind-map` - optional map of argument to bind declaration to declare the
    memory layout of the argument.
  * `:simple-mode?` - optional boolean to indicate not to perform optimizations
    that result in an unreadable AST.
  "
  ([fn-map {:keys [target-host]
            :or {target-host "llvm"}}]
   ;;map of target to IRModule

   (let [host-dev-modules (->> fn-map
                               (mapv (fn [[fn-name {:keys [schedule arguments bind-map simple-mode?
                                                           target]}]]
                                       (-> (lower schedule arguments {:name fn-name
                                                                      :binds bind-map
                                                                      :simple-mode? simple-mode?})
                                           (build-for-device (or target "llvm") target-host)))))
         host-modules (mapv first host-dev-modules)
         device-modules (vec (remove nil? (map second host-dev-modules)))
         host-module (let [host-module (ir-fns/IRModule {} {})]
                       (doseq [host-m host-modules]
                         (ir-fns/Module_Update host-module host-m))
                       (target-fns/Build host-module (target-fns/Target
                                                      (ast/safe-str target-host))))]
     (doseq [dev-mod device-modules]
       (TVMModImport host-module dev-mod))
     host-module))
  ([fn-map]
   (build fn-map nil)))





(comment
  (do
    (def n (ast/variable "n"))
    (def A (ast/placeholder [n] "A"))
    (def B (ast/placeholder [n] "B"))
    (def compute-op (ast/compute [n]
                                 (ast/tvm-fn
                                  [i]
                                  (ast/add (ast/tget A [i])
                                           (ast/tget B [i])))
                                 "C"))
    (def C (first (ast/output-tensors compute-op)))

    (def schedule (schedule/create-schedule compute-op))
    (schedule/stage-gpu-injective schedule compute-op))


  (def target (target-fns/Target "cuda"))
  (def input-mod (lower schedule [A B C] {:name "gpu_add"}))
  (def targetted-mod (map-pass #(assoc-attr % "target" target) input-mod))
  (def config (current-pass-context-config))

  (def mod-mixed (->> (concat
                       [(tir-transform-fns/VerifyMemory)]
                       (when (== 1 (count (:functions mod-mixed)))
                         [(make-fn-pass #(assoc-attr % "tir.is_entry_func" true))])
                       (when (get config "tir.detect_global_barrier")
                         [(tir-transform-fns/ThreadSync "global")])
                       [(tir-transform-fns/ThreadSync "shared")
                        (tir-transform-fns/ThreadSync "warp")
                        (tir-transform-fns/InferFragment)
                        (tir-transform-fns/LowerThreadAllreduce)
                        (tir-transform-fns/MakePackedAPI 0)
                        (tir-transform-fns/SplitHostDevice)])
                      (sequential-pass targetted-mod)))

  (def device-modules (->> [(make-filter-pass #(= (get (tvm-fn-attrs %) "calling_conv")
                                                  DEVICE_KERNEL_LAUNCH))
                            (tir-transform-fns/LowerWarpMemory)
                            (tir-transform-fns/Simplify)
                            (tir-transform-fns/LowerDeviceStorageAccessInfo)
                            (tir-transform-fns/LowerCustomDatatypes)
                            (tir-transform-fns/LowerIntrin)]
                           (sequential-pass mod-mixed)))

  (def host-modules (->> [(make-filter-pass #(not= (get (tvm-fn-attrs %) "calling_conv")
                                                   DEVICE_KERNEL_LAUNCH))
                          (make-fn-pass #(assoc-attr % "target" target))
                          (tir-transform-fns/LowerTVMBuiltin)
                          (tir-transform-fns/LowerDeviceStorageAccessInfo)
                          (tir-transform-fns/LowerCustomDatatypes)
                          (tir-transform-fns/LowerIntrin)
                          (tir-transform-fns/CombineContextCall)]
                         (sequential-pass mod-mixed)))

  (def module (build {"cpu_add" {:schedule schedule
                                 :arguments [A B C]}}))

  (def add-fn (module/get-module-function module "cpu_add"))
  (do

    (require '[tech.v3.tensor :as dtt])
    (def tens-a (dtt/->tensor (range 10) :datatype :float32 :container-type :native-heap))
    (def tens-b (dtt/->tensor (range 10 20) :datatype :float32 :container-type :native-heap))
    (def tens-c (dtt/new-tensor [10] :datatype :float32 :container-type :native-heap))
    (require '[tvm-clj.impl.dl-tensor])
    (add-fn tens-a tens-b tens-c)

    )
  )
