(ns tvm-clj.ast.elemwise-op
  "Elemwise TVM AST operators"
  (:require [tvm-clj.impl.protocols :as tvm-proto]
            [tvm-clj.impl.fns.tir :as tir-fns]
            [tvm-clj.impl.node :as jna-node]
            [tvm-clj.impl.base :as jna-base]
            [tech.v3.datatype :as dtype])
  (:import [com.sun.jna Pointer])
  (:refer-clojure :exclude [cast + - * / min max and or mod
                            > >= < <=]))


(defn const
  "Convert an item to a const (immediate) value"
  [numeric-value & [dtype]]
  (jna-node/const numeric-value dtype))


(defn cast
  "Cast a node to a different value."
  [expr-node dtype]
  (tir-fns/Cast (jna-node/->dtype dtype) expr-node nil))


(def ^:private call-types
  "Possible call types from Halide/IR.h"
  {:extern 0 ;;< A call to an external C-ABI function, possibly with side-effects
   :extern-c-plus-plus 1 ;;< A call to an external C-ABI function, possibly with side-effects
   :pure-extern 2 ;;< A call to a guaranteed-side-effect-free external function
   :halide 3 ;;< A call to a Func
   :intrinsic 4  ;;< A possibly-side-effecty compiler intrinsic, which has special handling during codegen
   :pure-intrinsic 5 ;;< A side-effect-free version of the above.
   })

(defn- ->call-type
  ^long [ctype]
  (cond
    (keyword? ctype)
    (if-let [retval (get call-types ctype)]
      retval
      (throw (ex-info "Failed to find call type"
                      {:call-type ctype})))
    (number? ctype)
    (long ctype)))


(def ^:private call-type-set (set (keys call-types)))


(defn- call
  "Call a 'function', which is basically executing a statement.  For instance, getting a
  value from the tensor is calling a halide function with the tensor's generating-op and
  value index."
  [ret-dtype fn-name fn-args call-type function-ref value-index]
  #_(bindings/global-node-function "make.Call" (->dtype ret-dtype) fn-name fn-args
                                   (->call-type call-type)
                                   function-ref value-index)
  (throw (Exception. "Failwhale")))


(defn- call-pure-intrin
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
  (call dtype func-name (tvm-proto/->node args) :pure-intrinsic nil 0))


(defn- call-intrin
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
  (call dtype func-name (tvm-proto/->node args) :intrinsic nil 0))


(defmacro ^:no-doc def-bin-op
  "Define a binary operation"
  [op-name op-fn]
  `(defn ~op-name
     [~'lhs ~'rhs]
     (~op-fn (tvm-proto/->node ~'lhs) (tvm-proto/->node ~'rhs) nil)))


(defmacro ^:no-doc def-op
  "Define a binary operation"
  [op-name make-name]
  `(defn ~op-name
     [~'lhs]
     (~op-name (tvm-proto/->node ~'lhs))))


(defmacro ^:no-doc def-bin-intrin-op
  [op-name]
  `(defn ~op-name
     [~'lhs ~'rhs]
     (call-pure-intrin (dtype/get-datatype ~'lhs)
                       ~(str op-name)
                       (tvm-proto/->node ~'lhs)
                       (tvm-proto/->node ~'rhs))))


(defmacro ^:no-doc def-intrin-op
  [op-name]
  `(defn ~op-name
     [~'lhs]
     (call-pure-intrin (dtype/get-datatype ~'lhs)
                       ~(str "tir." op-name)
                       (tvm-proto/->node ~'lhs))))


(def-bin-op + tir-fns/Add)
(def-bin-op - tir-fns/Sub)
(def-bin-op mod tir-fns/Mod)
(def-bin-op * tir-fns/Mul)
(def-bin-op / tir-fns/Div)
(def-bin-op eq tir-fns/_OpEQ)
(def-bin-op not-eq tir-fns/_OpNE)
(def-bin-op > tir-fns/_OpGT)
(def-bin-op >= tir-fns/_OpGE)
(def-bin-op < tir-fns/_OpLT)
(def-bin-op <= tir-fns/_OpLE)
(def-bin-op min tir-fns/_OpMin)


(defn min-value
  "Return an AST node that will generate the minimum value for a given datatype."
  [dtype]
  (let [[lval ltype] (jna-base/raw-call-function @tir-fns/min_value-fnptr*
                                                 (jna-node/->dtype dtype))]
    (jna-node/construct-node (Pointer. lval))))

(defn max-value
  "Return an AST node that will generate the maximum value for a given datatype."
  [dtype]
  (let [[lval ltype] (jna-base/raw-call-function @tir-fns/max_value-fnptr*
                                                 (jna-node/->dtype dtype))]
    (jna-node/construct-node (Pointer. lval))))

(def-bin-op max tir-fns/_OpMax)
(def-bin-op floor tir-fns/floor)
(def-bin-op ceil tir-fns/ceil)
(def-op abs tir-fns/abs)
(def-bin-op and tir-fns/And)
(def-bin-op or tir-fns/Or)
(def-intrin-op exp)
(def-intrin-op tanh)
(def-intrin-op sigmoid)
(def-intrin-op log)
(def-intrin-op sqrt)
(def-intrin-op trunc)
(def-op round tir-fns/round)
(def-bin-op pow tir-fns/_OpPow)


(defn select
  "Select between two expressions based on a condition.  Thus works similar to the
  clojure 'if' statement except it executes both branches.  This does not guard against
  out of bounds access; use if-then-else for that case.
  On the other hand, select can be vectorized while if-then-else cannot be."
  [bool-stmt true-stmt false-stmt]
  (tir-fns/Select (tvm-proto/->node bool-stmt)
                  (tvm-proto/->node true-stmt)
                  (tvm-proto/->node false-stmt)))


(defn if-then-else
  "Select between two expressions based on a condition.  Thus works similar to the
  clojure 'if' statement.  This is similar to 'select' except that it does not
  execute the wrong branch.  As a drawback, unlike select, it cannot be vectorized."
  [bool-stmt true-stmt false-stmt]
  (tir-fns/_OpIfThenElse (tvm-proto/->node bool-stmt)
                         (tvm-proto/->node true-stmt)
                         (tvm-proto/->node false-stmt)))
