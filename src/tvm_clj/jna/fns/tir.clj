(ns tvm-clj.jna.fns.tir
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Add
(let [gfn* (delay (jna-base/name->global-function "tir.Add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Allocate
(let [gfn* (delay (jna-base/name->global-function "tir.Allocate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} And
(let [gfn* (delay (jna-base/name->global-function "tir.And"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Any
(let [gfn* (delay (jna-base/name->global-function "tir.Any"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AssertStmt
(let [gfn* (delay (jna-base/name->global-function "tir.AssertStmt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} AttrStmt
(let [gfn* (delay (jna-base/name->global-function "tir.AttrStmt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BijectiveLayout
(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayout"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BijectiveLayoutBackwardIndex
(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutBackwardIndex"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BijectiveLayoutBackwardShape
(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutBackwardShape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BijectiveLayoutForwardIndex
(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutForwardIndex"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BijectiveLayoutForwardShape
(let [gfn* (delay (jna-base/name->global-function "tir.BijectiveLayoutForwardShape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Broadcast
(let [gfn* (delay (jna-base/name->global-function "tir.Broadcast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Buffer
(let [gfn* (delay (jna-base/name->global-function "tir.Buffer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BufferAccessPtr
(let [gfn* (delay (jna-base/name->global-function "tir.BufferAccessPtr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BufferLoad
(let [gfn* (delay (jna-base/name->global-function "tir.BufferLoad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BufferRealize
(let [gfn* (delay (jna-base/name->global-function "tir.BufferRealize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BufferStore
(let [gfn* (delay (jna-base/name->global-function "tir.BufferStore"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BufferVLoad
(let [gfn* (delay (jna-base/name->global-function "tir.BufferVLoad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} BufferVStore
(let [gfn* (delay (jna-base/name->global-function "tir.BufferVStore"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Call
(let [gfn* (delay (jna-base/name->global-function "tir.Call"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Cast
(let [gfn* (delay (jna-base/name->global-function "tir.Cast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CommReducer
(let [gfn* (delay (jna-base/name->global-function "tir.CommReducer"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} CommReducerCombine
(let [gfn* (delay (jna-base/name->global-function "tir.CommReducerCombine"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Div
(let [gfn* (delay (jna-base/name->global-function "tir.Div"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} EQ
(let [gfn* (delay (jna-base/name->global-function "tir.EQ"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Evaluate
(let [gfn* (delay (jna-base/name->global-function "tir.Evaluate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FloorDiv
(let [gfn* (delay (jna-base/name->global-function "tir.FloorDiv"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} FloorMod
(let [gfn* (delay (jna-base/name->global-function "tir.FloorMod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} For
(let [gfn* (delay (jna-base/name->global-function "tir.For"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GE
(let [gfn* (delay (jna-base/name->global-function "tir.GE"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} GT
(let [gfn* (delay (jna-base/name->global-function "tir.GT"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IRTransform
(let [gfn* (delay (jna-base/name->global-function "tir.IRTransform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IfThenElse
(let [gfn* (delay (jna-base/name->global-function "tir.IfThenElse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} IterVar
(let [gfn* (delay (jna-base/name->global-function "tir.IterVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LE
(let [gfn* (delay (jna-base/name->global-function "tir.LE"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LT
(let [gfn* (delay (jna-base/name->global-function "tir.LT"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Layout
(let [gfn* (delay (jna-base/name->global-function "tir.Layout"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LayoutFactorOf
(let [gfn* (delay (jna-base/name->global-function "tir.LayoutFactorOf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LayoutGetItem
(let [gfn* (delay (jna-base/name->global-function "tir.LayoutGetItem"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LayoutIndexOf
(let [gfn* (delay (jna-base/name->global-function "tir.LayoutIndexOf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LayoutNdim
(let [gfn* (delay (jna-base/name->global-function "tir.LayoutNdim"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Let
(let [gfn* (delay (jna-base/name->global-function "tir.Let"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} LetStmt
(let [gfn* (delay (jna-base/name->global-function "tir.LetStmt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Load
(let [gfn* (delay (jna-base/name->global-function "tir.Load"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Max
(let [gfn* (delay (jna-base/name->global-function "tir.Max"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Min
(let [gfn* (delay (jna-base/name->global-function "tir.Min"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Mod
(let [gfn* (delay (jna-base/name->global-function "tir.Mod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Mul
(let [gfn* (delay (jna-base/name->global-function "tir.Mul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} NE
(let [gfn* (delay (jna-base/name->global-function "tir.NE"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Not
(let [gfn* (delay (jna-base/name->global-function "tir.Not"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Or
(let [gfn* (delay (jna-base/name->global-function "tir.Or"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PostOrderVisit
(let [gfn* (delay (jna-base/name->global-function "tir.PostOrderVisit"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Prefetch
(let [gfn* (delay (jna-base/name->global-function "tir.Prefetch"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} PrimFunc
(let [gfn* (delay (jna-base/name->global-function "tir.PrimFunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ProducerLoad
(let [gfn* (delay (jna-base/name->global-function "tir.ProducerLoad"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ProducerRealize
(let [gfn* (delay (jna-base/name->global-function "tir.ProducerRealize"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ProducerStore
(let [gfn* (delay (jna-base/name->global-function "tir.ProducerStore"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Ramp
(let [gfn* (delay (jna-base/name->global-function "tir.Ramp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Reduce
(let [gfn* (delay (jna-base/name->global-function "tir.Reduce"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Select
(let [gfn* (delay (jna-base/name->global-function "tir.Select"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SeqStmt
(let [gfn* (delay (jna-base/name->global-function "tir.SeqStmt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Shuffle
(let [gfn* (delay (jna-base/name->global-function "tir.Shuffle"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} SizeVar
(let [gfn* (delay (jna-base/name->global-function "tir.SizeVar"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Store
(let [gfn* (delay (jna-base/name->global-function "tir.Store"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} StringImm
(let [gfn* (delay (jna-base/name->global-function "tir.StringImm"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Sub
(let [gfn* (delay (jna-base/name->global-function "tir.Sub"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Substitute
(let [gfn* (delay (jna-base/name->global-function "tir.Substitute"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} Var
(let [gfn* (delay (jna-base/name->global-function "tir.Var"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpAdd
(let [gfn* (delay (jna-base/name->global-function "tir._OpAdd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpAnd
(let [gfn* (delay (jna-base/name->global-function "tir._OpAnd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpDiv
(let [gfn* (delay (jna-base/name->global-function "tir._OpDiv"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpEQ
(let [gfn* (delay (jna-base/name->global-function "tir._OpEQ"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpFloorDiv
(let [gfn* (delay (jna-base/name->global-function "tir._OpFloorDiv"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpFloorMod
(let [gfn* (delay (jna-base/name->global-function "tir._OpFloorMod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpGE
(let [gfn* (delay (jna-base/name->global-function "tir._OpGE"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpGT
(let [gfn* (delay (jna-base/name->global-function "tir._OpGT"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpIfThenElse
(let [gfn* (delay (jna-base/name->global-function "tir._OpIfThenElse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpIndexDiv
(let [gfn* (delay (jna-base/name->global-function "tir._OpIndexDiv"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpIndexMod
(let [gfn* (delay (jna-base/name->global-function "tir._OpIndexMod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpLE
(let [gfn* (delay (jna-base/name->global-function "tir._OpLE"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpLT
(let [gfn* (delay (jna-base/name->global-function "tir._OpLT"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpMax
(let [gfn* (delay (jna-base/name->global-function "tir._OpMax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpMin
(let [gfn* (delay (jna-base/name->global-function "tir._OpMin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpMod
(let [gfn* (delay (jna-base/name->global-function "tir._OpMod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpMul
(let [gfn* (delay (jna-base/name->global-function "tir._OpMul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpNE
(let [gfn* (delay (jna-base/name->global-function "tir._OpNE"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpOr
(let [gfn* (delay (jna-base/name->global-function "tir._OpOr"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpPow
(let [gfn* (delay (jna-base/name->global-function "tir._OpPow"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpSub
(let [gfn* (delay (jna-base/name->global-function "tir._OpSub"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpTruncDiv
(let [gfn* (delay (jna-base/name->global-function "tir._OpTruncDiv"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _OpTruncMod
(let [gfn* (delay (jna-base/name->global-function "tir._OpTruncMod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _cast
(let [gfn* (delay (jna-base/name->global-function "tir._cast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} abs
(let [gfn* (delay (jna-base/name->global-function "tir.abs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_and
(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_and"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_not
(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_not"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_or
(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_or"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_xor
(let [gfn* (delay (jna-base/name->global-function "tir.bitwise_xor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ceil
(let [gfn* (delay (jna-base/name->global-function "tir.ceil"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor
(let [gfn* (delay (jna-base/name->global-function "tir.floor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isfinite
(let [gfn* (delay (jna-base/name->global-function "tir.isfinite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isinf
(let [gfn* (delay (jna-base/name->global-function "tir.isinf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isnan
(let [gfn* (delay (jna-base/name->global-function "tir.isnan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} left_shift
(let [gfn* (delay (jna-base/name->global-function "tir.left_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max_value
(let [gfn* (delay (jna-base/name->global-function "tir.max_value"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} min_value
(let [gfn* (delay (jna-base/name->global-function "tir.min_value"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} nearbyint
(let [gfn* (delay (jna-base/name->global-function "tir.nearbyint"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} right_shift
(let [gfn* (delay (jna-base/name->global-function "tir.right_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} round
(let [gfn* (delay (jna-base/name->global-function "tir.round"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} trunc
(let [gfn* (delay (jna-base/name->global-function "tir.trunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

