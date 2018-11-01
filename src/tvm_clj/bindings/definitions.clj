(ns tvm-clj.bindings.definitions
  (:require [clojure.set :as c-set]))


(def tvm-datatype->keyword-map
  {0 :int
   1 :uint
   2 :float
   3 :handle
   4 :null
   5 :tvm-type
   6 :tvm-context
   7 :array-handle
   8 :node-handle
   9 :module-handle
   10 :func-handle
   11 :string
   12 :bytes})


(def datatype->dl-type-code-map
  {:uint8 :uint
   :uint16 :uint
   :uint32 :uint
   :uint64 :uint
   :int8 :int
   :int16 :int
   :int32 :int
   :int64 :int
   :float32 :float
   :float64 :float})


(defn keyword->tvm-datatype
  [kwd]
  (if-let [retval (get (c-set/map-invert tvm-datatype->keyword-map) kwd)]
    retval
    (throw (ex-info "Failed to get tvm-datatype from kwd"
                    {:kwd kwd}))))


(defn tvm-datatype->keyword-nothrow
  [tvm-datatype]
  (get tvm-datatype->keyword-map tvm-datatype tvm-datatype))

(defn tvm-datatype->keyword
  [tvm-datatype]
  (if-let [retval (get tvm-datatype->keyword-map tvm-datatype)]
    retval
    (throw (ex-info "Failed to find keyword for tvm datatype"
                    {:tvm-datatype tvm-datatype}))))



(def dl-dtype-map->datatype-map
  {{:tvm-datatype :float
    :bits 32
    :lanes 1} :float32
   {:tvm-datatype :float
     :bits 64
    :lanes 1} :float64

   {:tvm-datatype :int
    :bits 8
    :lanes 1} :int8
   {:tvm-datatype :int
    :bits 16
    :lanes 1} :int16
   {:tvm-datatype :int
    :bits 32
    :lanes 1} :int32
   {:tvm-datatype :int
    :bits 64
    :lanes 1} :int64

   {:tvm-datatype :uint
    :bits 8
    :lanes 1} :uint8
   {:tvm-datatype :uint
    :bits 16
    :lanes 1} :uint16
   {:tvm-datatype :uint
    :bits 32
    :lanes 1} :uint32
   {:tvm-datatype :uint
    :bits 64
    :lanes 1} :uint64})


(def node-type-name->keyword-map
  {;;Container
   "Array" :array
   "Map" :map
   "Range" :range
   "LoweredFunc" :lowered-function
   ;;Expression
   "Expr" :expression
   "Variable" :variable
   "Reduce" :reduce
   "FloatImm" :float-imm
   "IntImm" :int-imm
   "UIntImm" :uint-imm
   "StringImm" :string-imm
   "Cast" :cast
   "Add" :+
   "Sub" :-
   "Mul" :*
   "Div" :/
   "Min" :min
   "Max" :max
   "EQ" :=
   "NE" :!=
   "LT" :<
   "LE" :<=
   "GT" :>
   "GE" :>=
   "And" :and
   "Not" :!
   "Select" :select
   "Load" :load
   "Ramp" :ramp
   "Broadcast" :broadcast
   "Shuffle" :shuffle
   "Call" :call
   "Let" :let
   ;;Schedule
   "Buffer" :buffer
   "Split" :split
   "Fuse" :fuse
   "IterVar" :iteration-variable
   "Schedule" :schedule
   "Stage" :stage
   ;;Tensor
   "Tensor" :tensor
   "PlaceholderOp" :placeholder-operation
   "ComputeOp" :compute-operation
   "ScanOp" :scan-operation
   "ExternOp" :external-operation
   ;;Statement
   "LetStmt" :let
   "AssertStmt" :assert
   "ProducerConsumer" :producer-consumer
   "For" :for
   "Store" :store
   "Provide" :provide
   "Allocate" :allocate
   "AttrStmt" :attribute
   "Free" :free
   "Realize" :realize
   "Block" :block
   "IfThenElse" :if-then-else
   "Evaluate" :evaluate
   "Prefetch" :prefetch
   ;;build-module
   "BuildConfig" :build-config
   ;;arith.py
   "IntervalSet" :interval-set
   "StrideSet" :stride-set
   "ModularSet" :modular-set
   })


(def expression-set
  #{:expression
    :variable
    :reduce
    :float-imm
    :int-imm
    :uint-imm
    :string-imm
    :cast
    :+
    :-
    :*
    :/
    :min
    :max
    :=
    :!=
    :<
    :<=
    :>
    :>=
    :and
    :!
    :select
    :load
    :ramp
    :broadcast
    :shuffle
    :call
    :let})


(defn is-expression-node?
  [node]
  (expression-set (:tvm-type-kwd node)))


(def kwd->device-type-map
  {:cpu 1
   :cpu-pinned 3
   :cuda 2
   :ext-dev 12
   :gpu 2
   :llvm 1
   :metal 8
   :opencl 4
   :rocm 10
   :stackvm 1
   :vpi 9
   :vulkan 7})

(def device-type->kwd-map (c-set/map-invert kwd->device-type-map))


(defn device-type->device-type-int
  ^long [device-type]
  (if-let [dev-enum (kwd->device-type-map device-type)]
    dev-enum
    (throw (ex-info "Failed to find device type enum"
                    {:device-type device-type}))))


(defn device-type-int->device-type
  [^long device-type]
  (if-let [retval (device-type->kwd-map device-type)]
    retval
    (throw (ex-info "Failed to find keyword for device type"
                    {:device-type device-type}))))


(def device-attribute-map
  {:exists 0
   :max-threads-per-block 1
   :warp-size 2
   :compute-version 3})
