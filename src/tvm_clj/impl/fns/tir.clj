(ns tvm-clj.jna.fns.tir
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "tir.Add"))]
  (defn Add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Allocate"))]
  (defn Allocate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Allocate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.And"))]
  (defn And
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.And"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Any"))]
  (defn Any
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Any"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.AssertStmt"))]
  (defn AssertStmt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.AssertStmt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.AttrStmt"))]
  (defn AttrStmt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.AttrStmt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayout"))]
  (defn BijectiveLayout
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BijectiveLayout"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutBackwardIndex"))]
  (defn BijectiveLayoutBackwardIndex
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BijectiveLayoutBackwardIndex"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutBackwardShape"))]
  (defn BijectiveLayoutBackwardShape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BijectiveLayoutBackwardShape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutForwardIndex"))]
  (defn BijectiveLayoutForwardIndex
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BijectiveLayoutForwardIndex"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutForwardShape"))]
  (defn BijectiveLayoutForwardShape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BijectiveLayoutForwardShape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Broadcast"))]
  (defn Broadcast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Broadcast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Buffer"))]
  (defn Buffer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Buffer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BufferAccessPtr"))]
  (defn BufferAccessPtr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BufferAccessPtr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BufferLoad"))]
  (defn BufferLoad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BufferLoad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BufferRealize"))]
  (defn BufferRealize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BufferRealize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BufferStore"))]
  (defn BufferStore
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BufferStore"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BufferVLoad"))]
  (defn BufferVLoad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BufferVLoad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.BufferVStore"))]
  (defn BufferVStore
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.BufferVStore"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Call"))]
  (defn Call
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Call"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Cast"))]
  (defn Cast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Cast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.CommReducer"))]
  (defn CommReducer
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.CommReducer"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.CommReducerCombine"))]
  (defn CommReducerCombine
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.CommReducerCombine"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Div"))]
  (defn Div
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Div"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.EQ"))]
  (defn EQ
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.EQ"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Evaluate"))]
  (defn Evaluate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Evaluate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.FloorDiv"))]
  (defn FloorDiv
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.FloorDiv"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.FloorMod"))]
  (defn FloorMod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.FloorMod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.For"))]
  (defn For
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.For"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.GE"))]
  (defn GE
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.GE"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.GT"))]
  (defn GT
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.GT"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.IRTransform"))]
  (defn IRTransform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.IRTransform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.IfThenElse"))]
  (defn IfThenElse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.IfThenElse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.IterVar"))]
  (defn IterVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.IterVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LE"))]
  (defn LE
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LE"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LT"))]
  (defn LT
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LT"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Layout"))]
  (defn Layout
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Layout"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LayoutFactorOf"))]
  (defn LayoutFactorOf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LayoutFactorOf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LayoutGetItem"))]
  (defn LayoutGetItem
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LayoutGetItem"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LayoutIndexOf"))]
  (defn LayoutIndexOf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LayoutIndexOf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LayoutNdim"))]
  (defn LayoutNdim
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LayoutNdim"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Let"))]
  (defn Let
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Let"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.LetStmt"))]
  (defn LetStmt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.LetStmt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Load"))]
  (defn Load
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Load"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Max"))]
  (defn Max
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Max"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Min"))]
  (defn Min
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Min"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Mod"))]
  (defn Mod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Mod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Mul"))]
  (defn Mul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Mul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.NE"))]
  (defn NE
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.NE"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Not"))]
  (defn Not
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Not"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Or"))]
  (defn Or
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Or"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.PostOrderVisit"))]
  (defn PostOrderVisit
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.PostOrderVisit"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Prefetch"))]
  (defn Prefetch
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Prefetch"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.PrimFunc"))]
  (defn PrimFunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.PrimFunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.ProducerLoad"))]
  (defn ProducerLoad
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.ProducerLoad"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.ProducerRealize"))]
  (defn ProducerRealize
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.ProducerRealize"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.ProducerStore"))]
  (defn ProducerStore
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.ProducerStore"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Ramp"))]
  (defn Ramp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Ramp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Reduce"))]
  (defn Reduce
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Reduce"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Select"))]
  (defn Select
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Select"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.SeqStmt"))]
  (defn SeqStmt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.SeqStmt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Shuffle"))]
  (defn Shuffle
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Shuffle"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.SizeVar"))]
  (defn SizeVar
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.SizeVar"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Store"))]
  (defn Store
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Store"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.StringImm"))]
  (defn StringImm
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.StringImm"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Sub"))]
  (defn Sub
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Sub"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Substitute"))]
  (defn Substitute
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Substitute"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.Var"))]
  (defn Var
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.Var"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpAdd"))]
  (defn _OpAdd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpAdd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpAnd"))]
  (defn _OpAnd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpAnd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpDiv"))]
  (defn _OpDiv
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpDiv"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpEQ"))]
  (defn _OpEQ
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpEQ"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpFloorDiv"))]
  (defn _OpFloorDiv
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpFloorDiv"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpFloorMod"))]
  (defn _OpFloorMod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpFloorMod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpGE"))]
  (defn _OpGE
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpGE"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpGT"))]
  (defn _OpGT
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpGT"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpIfThenElse"))]
  (defn _OpIfThenElse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpIfThenElse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpIndexDiv"))]
  (defn _OpIndexDiv
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpIndexDiv"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpIndexMod"))]
  (defn _OpIndexMod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpIndexMod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpLE"))]
  (defn _OpLE
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpLE"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpLT"))]
  (defn _OpLT
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpLT"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpMax"))]
  (defn _OpMax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpMax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpMin"))]
  (defn _OpMin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpMin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpMod"))]
  (defn _OpMod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpMod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpMul"))]
  (defn _OpMul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpMul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpNE"))]
  (defn _OpNE
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpNE"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpOr"))]
  (defn _OpOr
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpOr"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpPow"))]
  (defn _OpPow
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpPow"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpSub"))]
  (defn _OpSub
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpSub"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpTruncDiv"))]
  (defn _OpTruncDiv
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpTruncDiv"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._OpTruncMod"))]
  (defn _OpTruncMod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._OpTruncMod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir._cast"))]
  (defn _cast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir._cast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.abs"))]
  (defn abs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.abs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_and"))]
  (defn bitwise_and
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.bitwise_and"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_not"))]
  (defn bitwise_not
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.bitwise_not"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_or"))]
  (defn bitwise_or
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.bitwise_or"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_xor"))]
  (defn bitwise_xor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.bitwise_xor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.isfinite"))]
  (defn isfinite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.isfinite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.isinf"))]
  (defn isinf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.isinf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.isnan"))]
  (defn isnan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.isnan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.left_shift"))]
  (defn left_shift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.left_shift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.max_value"))]
  (defn max_value
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.max_value"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.min_value"))]
  (defn min_value
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.min_value"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.nearbyint"))]
  (defn nearbyint
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.nearbyint"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.right_shift"))]
  (defn right_shift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.right_shift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "tir.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "tir.trunc"}
     (apply jna-base/call-function @gfn* args))))

