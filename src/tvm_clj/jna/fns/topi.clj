(ns tvm-clj.jna.fns.topi
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} TEST_create_target
(let [gfn* (delay (jna-base/name->global-function "topi.TEST_create_target"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} acos
(let [gfn* (delay (jna-base/name->global-function "topi.acos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} acosh
(let [gfn* (delay (jna-base/name->global-function "topi.acosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} add
(let [gfn* (delay (jna-base/name->global-function "topi.add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adv_index
(let [gfn* (delay (jna-base/name->global-function "topi.adv_index"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} all
(let [gfn* (delay (jna-base/name->global-function "topi.all"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} any
(let [gfn* (delay (jna-base/name->global-function "topi.any"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} arange
(let [gfn* (delay (jna-base/name->global-function "topi.arange"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argmax
(let [gfn* (delay (jna-base/name->global-function "topi.argmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argmin
(let [gfn* (delay (jna-base/name->global-function "topi.argmin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} asin
(let [gfn* (delay (jna-base/name->global-function "topi.asin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} asinh
(let [gfn* (delay (jna-base/name->global-function "topi.asinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atan
(let [gfn* (delay (jna-base/name->global-function "topi.atan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atanh
(let [gfn* (delay (jna-base/name->global-function "topi.atanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_and
(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_and"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_not
(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_not"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_or
(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_or"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_xor
(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_xor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} broadcast_to
(let [gfn* (delay (jna-base/name->global-function "topi.broadcast_to"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cast
(let [gfn* (delay (jna-base/name->global-function "topi.cast"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} clip
(let [gfn* (delay (jna-base/name->global-function "topi.clip"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} concatenate
(let [gfn* (delay (jna-base/name->global-function "topi.concatenate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cos
(let [gfn* (delay (jna-base/name->global-function "topi.cos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cosh
(let [gfn* (delay (jna-base/name->global-function "topi.cosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} divide
(let [gfn* (delay (jna-base/name->global-function "topi.divide"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} elemwise_sum
(let [gfn* (delay (jna-base/name->global-function "topi.elemwise_sum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} equal
(let [gfn* (delay (jna-base/name->global-function "topi.equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} erf
(let [gfn* (delay (jna-base/name->global-function "topi.erf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp
(let [gfn* (delay (jna-base/name->global-function "topi.exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} expand_dims
(let [gfn* (delay (jna-base/name->global-function "topi.expand_dims"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fast_erf
(let [gfn* (delay (jna-base/name->global-function "topi.fast_erf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fast_exp
(let [gfn* (delay (jna-base/name->global-function "topi.fast_exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fast_tanh
(let [gfn* (delay (jna-base/name->global-function "topi.fast_tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} flip
(let [gfn* (delay (jna-base/name->global-function "topi.flip"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor_divide
(let [gfn* (delay (jna-base/name->global-function "topi.floor_divide"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor_mod
(let [gfn* (delay (jna-base/name->global-function "topi.floor_mod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} full
(let [gfn* (delay (jna-base/name->global-function "topi.full"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} full_like
(let [gfn* (delay (jna-base/name->global-function "topi.full_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} gather
(let [gfn* (delay (jna-base/name->global-function "topi.gather"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} gather_nd
(let [gfn* (delay (jna-base/name->global-function "topi.gather_nd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} greater
(let [gfn* (delay (jna-base/name->global-function "topi.greater"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} greater_equal
(let [gfn* (delay (jna-base/name->global-function "topi.greater_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} identity
(let [gfn* (delay (jna-base/name->global-function "topi.identity"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} layout_transform
(let [gfn* (delay (jna-base/name->global-function "topi.layout_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} left_shift
(let [gfn* (delay (jna-base/name->global-function "topi.left_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} less
(let [gfn* (delay (jna-base/name->global-function "topi.less"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} less_equal
(let [gfn* (delay (jna-base/name->global-function "topi.less_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log
(let [gfn* (delay (jna-base/name->global-function "topi.log"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log10
(let [gfn* (delay (jna-base/name->global-function "topi.log10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log2
(let [gfn* (delay (jna-base/name->global-function "topi.log2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_and
(let [gfn* (delay (jna-base/name->global-function "topi.logical_and"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_not
(let [gfn* (delay (jna-base/name->global-function "topi.logical_not"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_or
(let [gfn* (delay (jna-base/name->global-function "topi.logical_or"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_xor
(let [gfn* (delay (jna-base/name->global-function "topi.logical_xor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} matmul
(let [gfn* (delay (jna-base/name->global-function "topi.matmul"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} matrix_set_diag
(let [gfn* (delay (jna-base/name->global-function "topi.matrix_set_diag"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max
(let [gfn* (delay (jna-base/name->global-function "topi.max"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} maximum
(let [gfn* (delay (jna-base/name->global-function "topi.maximum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} meshgrid
(let [gfn* (delay (jna-base/name->global-function "topi.meshgrid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} min
(let [gfn* (delay (jna-base/name->global-function "topi.min"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} minimum
(let [gfn* (delay (jna-base/name->global-function "topi.minimum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} mod
(let [gfn* (delay (jna-base/name->global-function "topi.mod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} multiply
(let [gfn* (delay (jna-base/name->global-function "topi.multiply"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ndarray_size
(let [gfn* (delay (jna-base/name->global-function "topi.ndarray_size"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} negative
(let [gfn* (delay (jna-base/name->global-function "topi.negative"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} not_equal
(let [gfn* (delay (jna-base/name->global-function "topi.not_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} one_hot
(let [gfn* (delay (jna-base/name->global-function "topi.one_hot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} power
(let [gfn* (delay (jna-base/name->global-function "topi.power"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} prod
(let [gfn* (delay (jna-base/name->global-function "topi.prod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reinterpret
(let [gfn* (delay (jna-base/name->global-function "topi.reinterpret"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} repeat
(let [gfn* (delay (jna-base/name->global-function "topi.repeat"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reshape
(let [gfn* (delay (jna-base/name->global-function "topi.reshape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reverse_sequence
(let [gfn* (delay (jna-base/name->global-function "topi.reverse_sequence"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} right_shift
(let [gfn* (delay (jna-base/name->global-function "topi.right_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} rsqrt
(let [gfn* (delay (jna-base/name->global-function "topi.rsqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sequence_mask
(let [gfn* (delay (jna-base/name->global-function "topi.sequence_mask"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} shape
(let [gfn* (delay (jna-base/name->global-function "topi.shape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sigmoid
(let [gfn* (delay (jna-base/name->global-function "topi.sigmoid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sign
(let [gfn* (delay (jna-base/name->global-function "topi.sign"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sin
(let [gfn* (delay (jna-base/name->global-function "topi.sin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sinh
(let [gfn* (delay (jna-base/name->global-function "topi.sinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sparse_to_dense
(let [gfn* (delay (jna-base/name->global-function "topi.sparse_to_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} split
(let [gfn* (delay (jna-base/name->global-function "topi.split"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sqrt
(let [gfn* (delay (jna-base/name->global-function "topi.sqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} squeeze
(let [gfn* (delay (jna-base/name->global-function "topi.squeeze"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} stack
(let [gfn* (delay (jna-base/name->global-function "topi.stack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} strided_slice
(let [gfn* (delay (jna-base/name->global-function "topi.strided_slice"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} subtract
(let [gfn* (delay (jna-base/name->global-function "topi.subtract"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sum
(let [gfn* (delay (jna-base/name->global-function "topi.sum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} take
(let [gfn* (delay (jna-base/name->global-function "topi.take"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tan
(let [gfn* (delay (jna-base/name->global-function "topi.tan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tanh
(let [gfn* (delay (jna-base/name->global-function "topi.tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tensordot
(let [gfn* (delay (jna-base/name->global-function "topi.tensordot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tile
(let [gfn* (delay (jna-base/name->global-function "topi.tile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} transpose
(let [gfn* (delay (jna-base/name->global-function "topi.transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} unravel_index
(let [gfn* (delay (jna-base/name->global-function "topi.unravel_index"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} where
(let [gfn* (delay (jna-base/name->global-function "topi.where"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

