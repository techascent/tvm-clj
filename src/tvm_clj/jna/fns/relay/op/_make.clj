(ns tvm-clj.jna.fns.relay.op._make
  (:require [tvm-clj.jna.base :as jna-base]))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} OpStrategy
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.OpStrategy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} _variance
(let [gfn* (delay (jna-base/name->global-function "relay.op._make._variance"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} abs
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.abs"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} acos
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.acos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} acosh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.acosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} add
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} adv_index
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.adv_index"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} all
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.all"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} any
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.any"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} arange
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.arange"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argmax
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argmax"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argmin
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argmin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argsort
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argsort"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} argwhere
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argwhere"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} asin
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.asin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} asinh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.asinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atan
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.atan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} atanh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.atanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_and
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_and"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_not
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_not"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_or
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_or"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} bitwise_xor
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_xor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} broadcast_to
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.broadcast_to"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} broadcast_to_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.broadcast_to_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ceil
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ceil"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} clip
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.clip"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} collapse_sum_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.collapse_sum_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} collapse_sum_to
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.collapse_sum_to"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} concatenate
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.concatenate"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} contrib_reverse_reshape
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.contrib_reverse_reshape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} copy
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.copy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cos
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.cos"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} cosh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.cosh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} debug
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.debug"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} device_copy
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.device_copy"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} divide
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.divide"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} equal
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} erf
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.erf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} exp
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} expand_dims
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.expand_dims"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fast_erf
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fast_erf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fast_exp
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fast_exp"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fast_tanh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fast_tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} fixed_point_multiply
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fixed_point_multiply"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.floor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor_divide
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.floor_divide"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} floor_mod
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.floor_mod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} full
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.full"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} full_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.full_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} gather
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.gather"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} gather_nd
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.gather_nd"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} greater
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.greater"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} greater_equal
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.greater_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isfinite
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.isfinite"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isinf
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.isinf"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} isnan
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.isnan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} layout_transform
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.layout_transform"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} left_shift
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.left_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} less
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.less"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} less_equal
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.less_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.log"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log10
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.log10"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} log2
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.log2"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_and
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_and"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_not
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_not"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_or
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_or"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} logical_xor
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_xor"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} matrix_set_diag
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.matrix_set_diag"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} max
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.max"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} maximum
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.maximum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} mean
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.mean"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} meshgrid
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.meshgrid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} min
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.min"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} minimum
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.minimum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} mod
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.mod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} multiply
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.multiply"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ndarray_size
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ndarray_size"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} negative
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.negative"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} not_equal
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.not_equal"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} one_hot
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.one_hot"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ones
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ones"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} ones_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ones_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} power
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.power"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} prod
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.prod"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} repeat
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.repeat"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reshape
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reshape"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reshape_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reshape_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reverse
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reverse"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} reverse_sequence
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reverse_sequence"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} right_shift
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.right_shift"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} round
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.round"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} rsqrt
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.rsqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} scatter
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.scatter"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} scatter_add
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.scatter_add"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sequence_mask
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sequence_mask"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} shape_of
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.shape_of"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sigmoid
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sigmoid"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sign
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sign"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sin
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sin"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sinh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sinh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} slice_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.slice_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sparse_to_dense
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sparse_to_dense"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} split
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.split"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sqrt
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sqrt"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} squeeze
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.squeeze"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} stack
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.stack"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} strided_set
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.strided_set"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} strided_slice
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.strided_slice"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} subtract
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.subtract"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} sum
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sum"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} take
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.take"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tan
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.tan"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tanh
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.tanh"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} tile
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.tile"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} topk
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.topk"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} transpose
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.transpose"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} trunc
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.trunc"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} unravel_index
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.unravel_index"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} where
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.where"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} zeros
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.zeros"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

(def ^{:doc "TVM PackedFn"
:arglists '([& args])} zeros_like
(let [gfn* (delay (jna-base/name->global-function "relay.op._make.zeros_like"))]
    (fn [& args] (apply jna-base/call-function @gfn* args))))

