(ns tvm-clj.impl.fns.tir
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private Add-fnptr* (delay (base/name->global-function "tir.Add")))
(defn Add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Add"}
   (apply base/call-function @Add-fnptr* args)))

(defonce ^:private Allocate-fnptr* (delay (base/name->global-function "tir.Allocate")))
(defn Allocate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Allocate"}
   (apply base/call-function @Allocate-fnptr* args)))

(defonce ^:private And-fnptr* (delay (base/name->global-function "tir.And")))
(defn And
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.And"}
   (apply base/call-function @And-fnptr* args)))

(defonce ^:private Any-fnptr* (delay (base/name->global-function "tir.Any")))
(defn Any
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Any"}
   (apply base/call-function @Any-fnptr* args)))

(defonce ^:private AssertStmt-fnptr* (delay (base/name->global-function "tir.AssertStmt")))
(defn AssertStmt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.AssertStmt"}
   (apply base/call-function @AssertStmt-fnptr* args)))

(defonce ^:private AttrStmt-fnptr* (delay (base/name->global-function "tir.AttrStmt")))
(defn AttrStmt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.AttrStmt"}
   (apply base/call-function @AttrStmt-fnptr* args)))

(defonce ^:private BijectiveLayout-fnptr* (delay (base/name->global-function "tir.BijectiveLayout")))
(defn BijectiveLayout
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BijectiveLayout"}
   (apply base/call-function @BijectiveLayout-fnptr* args)))

(defonce ^:private BijectiveLayoutBackwardIndex-fnptr* (delay (base/name->global-function "tir.BijectiveLayoutBackwardIndex")))
(defn BijectiveLayoutBackwardIndex
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BijectiveLayoutBackwardIndex"}
   (apply base/call-function @BijectiveLayoutBackwardIndex-fnptr* args)))

(defonce ^:private BijectiveLayoutBackwardShape-fnptr* (delay (base/name->global-function "tir.BijectiveLayoutBackwardShape")))
(defn BijectiveLayoutBackwardShape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BijectiveLayoutBackwardShape"}
   (apply base/call-function @BijectiveLayoutBackwardShape-fnptr* args)))

(defonce ^:private BijectiveLayoutForwardIndex-fnptr* (delay (base/name->global-function "tir.BijectiveLayoutForwardIndex")))
(defn BijectiveLayoutForwardIndex
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BijectiveLayoutForwardIndex"}
   (apply base/call-function @BijectiveLayoutForwardIndex-fnptr* args)))

(defonce ^:private BijectiveLayoutForwardShape-fnptr* (delay (base/name->global-function "tir.BijectiveLayoutForwardShape")))
(defn BijectiveLayoutForwardShape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BijectiveLayoutForwardShape"}
   (apply base/call-function @BijectiveLayoutForwardShape-fnptr* args)))

(defonce ^:private Broadcast-fnptr* (delay (base/name->global-function "tir.Broadcast")))
(defn Broadcast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Broadcast"}
   (apply base/call-function @Broadcast-fnptr* args)))

(defonce ^:private Buffer-fnptr* (delay (base/name->global-function "tir.Buffer")))
(defn Buffer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Buffer"}
   (apply base/call-function @Buffer-fnptr* args)))

(defonce ^:private BufferAccessPtr-fnptr* (delay (base/name->global-function "tir.BufferAccessPtr")))
(defn BufferAccessPtr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BufferAccessPtr"}
   (apply base/call-function @BufferAccessPtr-fnptr* args)))

(defonce ^:private BufferLoad-fnptr* (delay (base/name->global-function "tir.BufferLoad")))
(defn BufferLoad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BufferLoad"}
   (apply base/call-function @BufferLoad-fnptr* args)))

(defonce ^:private BufferRealize-fnptr* (delay (base/name->global-function "tir.BufferRealize")))
(defn BufferRealize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BufferRealize"}
   (apply base/call-function @BufferRealize-fnptr* args)))

(defonce ^:private BufferStore-fnptr* (delay (base/name->global-function "tir.BufferStore")))
(defn BufferStore
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BufferStore"}
   (apply base/call-function @BufferStore-fnptr* args)))

(defonce ^:private BufferVLoad-fnptr* (delay (base/name->global-function "tir.BufferVLoad")))
(defn BufferVLoad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BufferVLoad"}
   (apply base/call-function @BufferVLoad-fnptr* args)))

(defonce ^:private BufferVStore-fnptr* (delay (base/name->global-function "tir.BufferVStore")))
(defn BufferVStore
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.BufferVStore"}
   (apply base/call-function @BufferVStore-fnptr* args)))

(defonce ^:private Call-fnptr* (delay (base/name->global-function "tir.Call")))
(defn Call
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Call"}
   (apply base/call-function @Call-fnptr* args)))

(defonce ^:private Cast-fnptr* (delay (base/name->global-function "tir.Cast")))
(defn Cast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Cast"}
   (apply base/call-function @Cast-fnptr* args)))

(defonce ^:private CommReducer-fnptr* (delay (base/name->global-function "tir.CommReducer")))
(defn CommReducer
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.CommReducer"}
   (apply base/call-function @CommReducer-fnptr* args)))

(defonce ^:private CommReducerCombine-fnptr* (delay (base/name->global-function "tir.CommReducerCombine")))
(defn CommReducerCombine
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.CommReducerCombine"}
   (apply base/call-function @CommReducerCombine-fnptr* args)))

(defonce ^:private Div-fnptr* (delay (base/name->global-function "tir.Div")))
(defn Div
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Div"}
   (apply base/call-function @Div-fnptr* args)))

(defonce ^:private EQ-fnptr* (delay (base/name->global-function "tir.EQ")))
(defn EQ
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.EQ"}
   (apply base/call-function @EQ-fnptr* args)))

(defonce ^:private Evaluate-fnptr* (delay (base/name->global-function "tir.Evaluate")))
(defn Evaluate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Evaluate"}
   (apply base/call-function @Evaluate-fnptr* args)))

(defonce ^:private FloorDiv-fnptr* (delay (base/name->global-function "tir.FloorDiv")))
(defn FloorDiv
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.FloorDiv"}
   (apply base/call-function @FloorDiv-fnptr* args)))

(defonce ^:private FloorMod-fnptr* (delay (base/name->global-function "tir.FloorMod")))
(defn FloorMod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.FloorMod"}
   (apply base/call-function @FloorMod-fnptr* args)))

(defonce ^:private For-fnptr* (delay (base/name->global-function "tir.For")))
(defn For
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.For"}
   (apply base/call-function @For-fnptr* args)))

(defonce ^:private GE-fnptr* (delay (base/name->global-function "tir.GE")))
(defn GE
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.GE"}
   (apply base/call-function @GE-fnptr* args)))

(defonce ^:private GT-fnptr* (delay (base/name->global-function "tir.GT")))
(defn GT
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.GT"}
   (apply base/call-function @GT-fnptr* args)))

(defonce ^:private IRTransform-fnptr* (delay (base/name->global-function "tir.IRTransform")))
(defn IRTransform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.IRTransform"}
   (apply base/call-function @IRTransform-fnptr* args)))

(defonce ^:private IfThenElse-fnptr* (delay (base/name->global-function "tir.IfThenElse")))
(defn IfThenElse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.IfThenElse"}
   (apply base/call-function @IfThenElse-fnptr* args)))

(defonce ^:private IterVar-fnptr* (delay (base/name->global-function "tir.IterVar")))
(defn IterVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.IterVar"}
   (apply base/call-function @IterVar-fnptr* args)))

(defonce ^:private LE-fnptr* (delay (base/name->global-function "tir.LE")))
(defn LE
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LE"}
   (apply base/call-function @LE-fnptr* args)))

(defonce ^:private LT-fnptr* (delay (base/name->global-function "tir.LT")))
(defn LT
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LT"}
   (apply base/call-function @LT-fnptr* args)))

(defonce ^:private Layout-fnptr* (delay (base/name->global-function "tir.Layout")))
(defn Layout
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Layout"}
   (apply base/call-function @Layout-fnptr* args)))

(defonce ^:private LayoutFactorOf-fnptr* (delay (base/name->global-function "tir.LayoutFactorOf")))
(defn LayoutFactorOf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LayoutFactorOf"}
   (apply base/call-function @LayoutFactorOf-fnptr* args)))

(defonce ^:private LayoutGetItem-fnptr* (delay (base/name->global-function "tir.LayoutGetItem")))
(defn LayoutGetItem
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LayoutGetItem"}
   (apply base/call-function @LayoutGetItem-fnptr* args)))

(defonce ^:private LayoutIndexOf-fnptr* (delay (base/name->global-function "tir.LayoutIndexOf")))
(defn LayoutIndexOf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LayoutIndexOf"}
   (apply base/call-function @LayoutIndexOf-fnptr* args)))

(defonce ^:private LayoutNdim-fnptr* (delay (base/name->global-function "tir.LayoutNdim")))
(defn LayoutNdim
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LayoutNdim"}
   (apply base/call-function @LayoutNdim-fnptr* args)))

(defonce ^:private Let-fnptr* (delay (base/name->global-function "tir.Let")))
(defn Let
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Let"}
   (apply base/call-function @Let-fnptr* args)))

(defonce ^:private LetStmt-fnptr* (delay (base/name->global-function "tir.LetStmt")))
(defn LetStmt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.LetStmt"}
   (apply base/call-function @LetStmt-fnptr* args)))

(defonce ^:private Load-fnptr* (delay (base/name->global-function "tir.Load")))
(defn Load
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Load"}
   (apply base/call-function @Load-fnptr* args)))

(defonce ^:private Max-fnptr* (delay (base/name->global-function "tir.Max")))
(defn Max
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Max"}
   (apply base/call-function @Max-fnptr* args)))

(defonce ^:private Min-fnptr* (delay (base/name->global-function "tir.Min")))
(defn Min
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Min"}
   (apply base/call-function @Min-fnptr* args)))

(defonce ^:private Mod-fnptr* (delay (base/name->global-function "tir.Mod")))
(defn Mod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Mod"}
   (apply base/call-function @Mod-fnptr* args)))

(defonce ^:private Mul-fnptr* (delay (base/name->global-function "tir.Mul")))
(defn Mul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Mul"}
   (apply base/call-function @Mul-fnptr* args)))

(defonce ^:private NE-fnptr* (delay (base/name->global-function "tir.NE")))
(defn NE
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.NE"}
   (apply base/call-function @NE-fnptr* args)))

(defonce ^:private Not-fnptr* (delay (base/name->global-function "tir.Not")))
(defn Not
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Not"}
   (apply base/call-function @Not-fnptr* args)))

(defonce ^:private Or-fnptr* (delay (base/name->global-function "tir.Or")))
(defn Or
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Or"}
   (apply base/call-function @Or-fnptr* args)))

(defonce ^:private PostOrderVisit-fnptr* (delay (base/name->global-function "tir.PostOrderVisit")))
(defn PostOrderVisit
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.PostOrderVisit"}
   (apply base/call-function @PostOrderVisit-fnptr* args)))

(defonce ^:private Prefetch-fnptr* (delay (base/name->global-function "tir.Prefetch")))
(defn Prefetch
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Prefetch"}
   (apply base/call-function @Prefetch-fnptr* args)))

(defonce ^:private PrimFunc-fnptr* (delay (base/name->global-function "tir.PrimFunc")))
(defn PrimFunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.PrimFunc"}
   (apply base/call-function @PrimFunc-fnptr* args)))

(defonce ^:private ProducerLoad-fnptr* (delay (base/name->global-function "tir.ProducerLoad")))
(defn ProducerLoad
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.ProducerLoad"}
   (apply base/call-function @ProducerLoad-fnptr* args)))

(defonce ^:private ProducerRealize-fnptr* (delay (base/name->global-function "tir.ProducerRealize")))
(defn ProducerRealize
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.ProducerRealize"}
   (apply base/call-function @ProducerRealize-fnptr* args)))

(defonce ^:private ProducerStore-fnptr* (delay (base/name->global-function "tir.ProducerStore")))
(defn ProducerStore
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.ProducerStore"}
   (apply base/call-function @ProducerStore-fnptr* args)))

(defonce ^:private Ramp-fnptr* (delay (base/name->global-function "tir.Ramp")))
(defn Ramp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Ramp"}
   (apply base/call-function @Ramp-fnptr* args)))

(defonce ^:private Reduce-fnptr* (delay (base/name->global-function "tir.Reduce")))
(defn Reduce
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Reduce"}
   (apply base/call-function @Reduce-fnptr* args)))

(defonce ^:private Select-fnptr* (delay (base/name->global-function "tir.Select")))
(defn Select
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Select"}
   (apply base/call-function @Select-fnptr* args)))

(defonce ^:private SeqStmt-fnptr* (delay (base/name->global-function "tir.SeqStmt")))
(defn SeqStmt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.SeqStmt"}
   (apply base/call-function @SeqStmt-fnptr* args)))

(defonce ^:private Shuffle-fnptr* (delay (base/name->global-function "tir.Shuffle")))
(defn Shuffle
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Shuffle"}
   (apply base/call-function @Shuffle-fnptr* args)))

(defonce ^:private SizeVar-fnptr* (delay (base/name->global-function "tir.SizeVar")))
(defn SizeVar
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.SizeVar"}
   (apply base/call-function @SizeVar-fnptr* args)))

(defonce ^:private Store-fnptr* (delay (base/name->global-function "tir.Store")))
(defn Store
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Store"}
   (apply base/call-function @Store-fnptr* args)))

(defonce ^:private StringImm-fnptr* (delay (base/name->global-function "tir.StringImm")))
(defn StringImm
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.StringImm"}
   (apply base/call-function @StringImm-fnptr* args)))

(defonce ^:private Sub-fnptr* (delay (base/name->global-function "tir.Sub")))
(defn Sub
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Sub"}
   (apply base/call-function @Sub-fnptr* args)))

(defonce ^:private Substitute-fnptr* (delay (base/name->global-function "tir.Substitute")))
(defn Substitute
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Substitute"}
   (apply base/call-function @Substitute-fnptr* args)))

(defonce ^:private Var-fnptr* (delay (base/name->global-function "tir.Var")))
(defn Var
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.Var"}
   (apply base/call-function @Var-fnptr* args)))

(defonce ^:private _OpAdd-fnptr* (delay (base/name->global-function "tir._OpAdd")))
(defn _OpAdd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpAdd"}
   (apply base/call-function @_OpAdd-fnptr* args)))

(defonce ^:private _OpAnd-fnptr* (delay (base/name->global-function "tir._OpAnd")))
(defn _OpAnd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpAnd"}
   (apply base/call-function @_OpAnd-fnptr* args)))

(defonce ^:private _OpDiv-fnptr* (delay (base/name->global-function "tir._OpDiv")))
(defn _OpDiv
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpDiv"}
   (apply base/call-function @_OpDiv-fnptr* args)))

(defonce ^:private _OpEQ-fnptr* (delay (base/name->global-function "tir._OpEQ")))
(defn _OpEQ
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpEQ"}
   (apply base/call-function @_OpEQ-fnptr* args)))

(defonce ^:private _OpFloorDiv-fnptr* (delay (base/name->global-function "tir._OpFloorDiv")))
(defn _OpFloorDiv
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpFloorDiv"}
   (apply base/call-function @_OpFloorDiv-fnptr* args)))

(defonce ^:private _OpFloorMod-fnptr* (delay (base/name->global-function "tir._OpFloorMod")))
(defn _OpFloorMod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpFloorMod"}
   (apply base/call-function @_OpFloorMod-fnptr* args)))

(defonce ^:private _OpGE-fnptr* (delay (base/name->global-function "tir._OpGE")))
(defn _OpGE
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpGE"}
   (apply base/call-function @_OpGE-fnptr* args)))

(defonce ^:private _OpGT-fnptr* (delay (base/name->global-function "tir._OpGT")))
(defn _OpGT
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpGT"}
   (apply base/call-function @_OpGT-fnptr* args)))

(defonce ^:private _OpIfThenElse-fnptr* (delay (base/name->global-function "tir._OpIfThenElse")))
(defn _OpIfThenElse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpIfThenElse"}
   (apply base/call-function @_OpIfThenElse-fnptr* args)))

(defonce ^:private _OpIndexDiv-fnptr* (delay (base/name->global-function "tir._OpIndexDiv")))
(defn _OpIndexDiv
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpIndexDiv"}
   (apply base/call-function @_OpIndexDiv-fnptr* args)))

(defonce ^:private _OpIndexMod-fnptr* (delay (base/name->global-function "tir._OpIndexMod")))
(defn _OpIndexMod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpIndexMod"}
   (apply base/call-function @_OpIndexMod-fnptr* args)))

(defonce ^:private _OpLE-fnptr* (delay (base/name->global-function "tir._OpLE")))
(defn _OpLE
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpLE"}
   (apply base/call-function @_OpLE-fnptr* args)))

(defonce ^:private _OpLT-fnptr* (delay (base/name->global-function "tir._OpLT")))
(defn _OpLT
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpLT"}
   (apply base/call-function @_OpLT-fnptr* args)))

(defonce ^:private _OpMax-fnptr* (delay (base/name->global-function "tir._OpMax")))
(defn _OpMax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpMax"}
   (apply base/call-function @_OpMax-fnptr* args)))

(defonce ^:private _OpMin-fnptr* (delay (base/name->global-function "tir._OpMin")))
(defn _OpMin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpMin"}
   (apply base/call-function @_OpMin-fnptr* args)))

(defonce ^:private _OpMod-fnptr* (delay (base/name->global-function "tir._OpMod")))
(defn _OpMod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpMod"}
   (apply base/call-function @_OpMod-fnptr* args)))

(defonce ^:private _OpMul-fnptr* (delay (base/name->global-function "tir._OpMul")))
(defn _OpMul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpMul"}
   (apply base/call-function @_OpMul-fnptr* args)))

(defonce ^:private _OpNE-fnptr* (delay (base/name->global-function "tir._OpNE")))
(defn _OpNE
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpNE"}
   (apply base/call-function @_OpNE-fnptr* args)))

(defonce ^:private _OpOr-fnptr* (delay (base/name->global-function "tir._OpOr")))
(defn _OpOr
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpOr"}
   (apply base/call-function @_OpOr-fnptr* args)))

(defonce ^:private _OpPow-fnptr* (delay (base/name->global-function "tir._OpPow")))
(defn _OpPow
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpPow"}
   (apply base/call-function @_OpPow-fnptr* args)))

(defonce ^:private _OpSub-fnptr* (delay (base/name->global-function "tir._OpSub")))
(defn _OpSub
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpSub"}
   (apply base/call-function @_OpSub-fnptr* args)))

(defonce ^:private _OpTruncDiv-fnptr* (delay (base/name->global-function "tir._OpTruncDiv")))
(defn _OpTruncDiv
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpTruncDiv"}
   (apply base/call-function @_OpTruncDiv-fnptr* args)))

(defonce ^:private _OpTruncMod-fnptr* (delay (base/name->global-function "tir._OpTruncMod")))
(defn _OpTruncMod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._OpTruncMod"}
   (apply base/call-function @_OpTruncMod-fnptr* args)))

(defonce ^:private _cast-fnptr* (delay (base/name->global-function "tir._cast")))
(defn _cast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir._cast"}
   (apply base/call-function @_cast-fnptr* args)))

(defonce ^:private abs-fnptr* (delay (base/name->global-function "tir.abs")))
(defn abs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.abs"}
   (apply base/call-function @abs-fnptr* args)))

(defonce ^:private bitwise_and-fnptr* (delay (base/name->global-function "tir.bitwise_and")))
(defn bitwise_and
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.bitwise_and"}
   (apply base/call-function @bitwise_and-fnptr* args)))

(defonce ^:private bitwise_not-fnptr* (delay (base/name->global-function "tir.bitwise_not")))
(defn bitwise_not
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.bitwise_not"}
   (apply base/call-function @bitwise_not-fnptr* args)))

(defonce ^:private bitwise_or-fnptr* (delay (base/name->global-function "tir.bitwise_or")))
(defn bitwise_or
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.bitwise_or"}
   (apply base/call-function @bitwise_or-fnptr* args)))

(defonce ^:private bitwise_xor-fnptr* (delay (base/name->global-function "tir.bitwise_xor")))
(defn bitwise_xor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.bitwise_xor"}
   (apply base/call-function @bitwise_xor-fnptr* args)))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "tir.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "tir.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private isfinite-fnptr* (delay (base/name->global-function "tir.isfinite")))
(defn isfinite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.isfinite"}
   (apply base/call-function @isfinite-fnptr* args)))

(defonce ^:private isinf-fnptr* (delay (base/name->global-function "tir.isinf")))
(defn isinf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.isinf"}
   (apply base/call-function @isinf-fnptr* args)))

(defonce ^:private isnan-fnptr* (delay (base/name->global-function "tir.isnan")))
(defn isnan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.isnan"}
   (apply base/call-function @isnan-fnptr* args)))

(defonce ^:private left_shift-fnptr* (delay (base/name->global-function "tir.left_shift")))
(defn left_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.left_shift"}
   (apply base/call-function @left_shift-fnptr* args)))

(defonce ^:private max_value-fnptr* (delay (base/name->global-function "tir.max_value")))
(defn max_value
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.max_value"}
   (apply base/call-function @max_value-fnptr* args)))

(defonce ^:private min_value-fnptr* (delay (base/name->global-function "tir.min_value")))
(defn min_value
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.min_value"}
   (apply base/call-function @min_value-fnptr* args)))

(defonce ^:private nearbyint-fnptr* (delay (base/name->global-function "tir.nearbyint")))
(defn nearbyint
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.nearbyint"}
   (apply base/call-function @nearbyint-fnptr* args)))

(defonce ^:private right_shift-fnptr* (delay (base/name->global-function "tir.right_shift")))
(defn right_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.right_shift"}
   (apply base/call-function @right_shift-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "tir.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "tir.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "tir.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

