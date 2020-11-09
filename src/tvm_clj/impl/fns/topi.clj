(ns tvm-clj.impl.fns.topi
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private TEST_create_target-fnptr* (delay (base/name->global-function "topi.TEST_create_target")))
(defn TEST_create_target
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.TEST_create_target"}
   (apply base/call-function @TEST_create_target-fnptr* args)))

(defonce ^:private acos-fnptr* (delay (base/name->global-function "topi.acos")))
(defn acos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.acos"}
   (apply base/call-function @acos-fnptr* args)))

(defonce ^:private acosh-fnptr* (delay (base/name->global-function "topi.acosh")))
(defn acosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.acosh"}
   (apply base/call-function @acosh-fnptr* args)))

(defonce ^:private add-fnptr* (delay (base/name->global-function "topi.add")))
(defn add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.add"}
   (apply base/call-function @add-fnptr* args)))

(defonce ^:private adv_index-fnptr* (delay (base/name->global-function "topi.adv_index")))
(defn adv_index
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.adv_index"}
   (apply base/call-function @adv_index-fnptr* args)))

(defonce ^:private all-fnptr* (delay (base/name->global-function "topi.all")))
(defn all
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.all"}
   (apply base/call-function @all-fnptr* args)))

(defonce ^:private any-fnptr* (delay (base/name->global-function "topi.any")))
(defn any
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.any"}
   (apply base/call-function @any-fnptr* args)))

(defonce ^:private arange-fnptr* (delay (base/name->global-function "topi.arange")))
(defn arange
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.arange"}
   (apply base/call-function @arange-fnptr* args)))

(defonce ^:private argmax-fnptr* (delay (base/name->global-function "topi.argmax")))
(defn argmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.argmax"}
   (apply base/call-function @argmax-fnptr* args)))

(defonce ^:private argmin-fnptr* (delay (base/name->global-function "topi.argmin")))
(defn argmin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.argmin"}
   (apply base/call-function @argmin-fnptr* args)))

(defonce ^:private asin-fnptr* (delay (base/name->global-function "topi.asin")))
(defn asin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.asin"}
   (apply base/call-function @asin-fnptr* args)))

(defonce ^:private asinh-fnptr* (delay (base/name->global-function "topi.asinh")))
(defn asinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.asinh"}
   (apply base/call-function @asinh-fnptr* args)))

(defonce ^:private atan-fnptr* (delay (base/name->global-function "topi.atan")))
(defn atan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.atan"}
   (apply base/call-function @atan-fnptr* args)))

(defonce ^:private atanh-fnptr* (delay (base/name->global-function "topi.atanh")))
(defn atanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.atanh"}
   (apply base/call-function @atanh-fnptr* args)))

(defonce ^:private bitwise_and-fnptr* (delay (base/name->global-function "topi.bitwise_and")))
(defn bitwise_and
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.bitwise_and"}
   (apply base/call-function @bitwise_and-fnptr* args)))

(defonce ^:private bitwise_not-fnptr* (delay (base/name->global-function "topi.bitwise_not")))
(defn bitwise_not
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.bitwise_not"}
   (apply base/call-function @bitwise_not-fnptr* args)))

(defonce ^:private bitwise_or-fnptr* (delay (base/name->global-function "topi.bitwise_or")))
(defn bitwise_or
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.bitwise_or"}
   (apply base/call-function @bitwise_or-fnptr* args)))

(defonce ^:private bitwise_xor-fnptr* (delay (base/name->global-function "topi.bitwise_xor")))
(defn bitwise_xor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.bitwise_xor"}
   (apply base/call-function @bitwise_xor-fnptr* args)))

(defonce ^:private broadcast_to-fnptr* (delay (base/name->global-function "topi.broadcast_to")))
(defn broadcast_to
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.broadcast_to"}
   (apply base/call-function @broadcast_to-fnptr* args)))

(defonce ^:private cast-fnptr* (delay (base/name->global-function "topi.cast")))
(defn cast
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cast"}
   (apply base/call-function @cast-fnptr* args)))

(defonce ^:private clip-fnptr* (delay (base/name->global-function "topi.clip")))
(defn clip
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.clip"}
   (apply base/call-function @clip-fnptr* args)))

(defonce ^:private concatenate-fnptr* (delay (base/name->global-function "topi.concatenate")))
(defn concatenate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.concatenate"}
   (apply base/call-function @concatenate-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "topi.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "topi.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private divide-fnptr* (delay (base/name->global-function "topi.divide")))
(defn divide
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.divide"}
   (apply base/call-function @divide-fnptr* args)))

(defonce ^:private elemwise_sum-fnptr* (delay (base/name->global-function "topi.elemwise_sum")))
(defn elemwise_sum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.elemwise_sum"}
   (apply base/call-function @elemwise_sum-fnptr* args)))

(defonce ^:private equal-fnptr* (delay (base/name->global-function "topi.equal")))
(defn equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.equal"}
   (apply base/call-function @equal-fnptr* args)))

(defonce ^:private erf-fnptr* (delay (base/name->global-function "topi.erf")))
(defn erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.erf"}
   (apply base/call-function @erf-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "topi.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private expand_dims-fnptr* (delay (base/name->global-function "topi.expand_dims")))
(defn expand_dims
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.expand_dims"}
   (apply base/call-function @expand_dims-fnptr* args)))

(defonce ^:private fast_erf-fnptr* (delay (base/name->global-function "topi.fast_erf")))
(defn fast_erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.fast_erf"}
   (apply base/call-function @fast_erf-fnptr* args)))

(defonce ^:private fast_exp-fnptr* (delay (base/name->global-function "topi.fast_exp")))
(defn fast_exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.fast_exp"}
   (apply base/call-function @fast_exp-fnptr* args)))

(defonce ^:private fast_tanh-fnptr* (delay (base/name->global-function "topi.fast_tanh")))
(defn fast_tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.fast_tanh"}
   (apply base/call-function @fast_tanh-fnptr* args)))

(defonce ^:private flip-fnptr* (delay (base/name->global-function "topi.flip")))
(defn flip
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.flip"}
   (apply base/call-function @flip-fnptr* args)))

(defonce ^:private floor_divide-fnptr* (delay (base/name->global-function "topi.floor_divide")))
(defn floor_divide
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.floor_divide"}
   (apply base/call-function @floor_divide-fnptr* args)))

(defonce ^:private floor_mod-fnptr* (delay (base/name->global-function "topi.floor_mod")))
(defn floor_mod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.floor_mod"}
   (apply base/call-function @floor_mod-fnptr* args)))

(defonce ^:private full-fnptr* (delay (base/name->global-function "topi.full")))
(defn full
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.full"}
   (apply base/call-function @full-fnptr* args)))

(defonce ^:private full_like-fnptr* (delay (base/name->global-function "topi.full_like")))
(defn full_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.full_like"}
   (apply base/call-function @full_like-fnptr* args)))

(defonce ^:private gather-fnptr* (delay (base/name->global-function "topi.gather")))
(defn gather
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.gather"}
   (apply base/call-function @gather-fnptr* args)))

(defonce ^:private gather_nd-fnptr* (delay (base/name->global-function "topi.gather_nd")))
(defn gather_nd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.gather_nd"}
   (apply base/call-function @gather_nd-fnptr* args)))

(defonce ^:private greater-fnptr* (delay (base/name->global-function "topi.greater")))
(defn greater
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.greater"}
   (apply base/call-function @greater-fnptr* args)))

(defonce ^:private greater_equal-fnptr* (delay (base/name->global-function "topi.greater_equal")))
(defn greater_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.greater_equal"}
   (apply base/call-function @greater_equal-fnptr* args)))

(defonce ^:private identity-fnptr* (delay (base/name->global-function "topi.identity")))
(defn identity
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.identity"}
   (apply base/call-function @identity-fnptr* args)))

(defonce ^:private layout_transform-fnptr* (delay (base/name->global-function "topi.layout_transform")))
(defn layout_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.layout_transform"}
   (apply base/call-function @layout_transform-fnptr* args)))

(defonce ^:private left_shift-fnptr* (delay (base/name->global-function "topi.left_shift")))
(defn left_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.left_shift"}
   (apply base/call-function @left_shift-fnptr* args)))

(defonce ^:private less-fnptr* (delay (base/name->global-function "topi.less")))
(defn less
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.less"}
   (apply base/call-function @less-fnptr* args)))

(defonce ^:private less_equal-fnptr* (delay (base/name->global-function "topi.less_equal")))
(defn less_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.less_equal"}
   (apply base/call-function @less_equal-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "topi.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "topi.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "topi.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private logical_and-fnptr* (delay (base/name->global-function "topi.logical_and")))
(defn logical_and
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.logical_and"}
   (apply base/call-function @logical_and-fnptr* args)))

(defonce ^:private logical_not-fnptr* (delay (base/name->global-function "topi.logical_not")))
(defn logical_not
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.logical_not"}
   (apply base/call-function @logical_not-fnptr* args)))

(defonce ^:private logical_or-fnptr* (delay (base/name->global-function "topi.logical_or")))
(defn logical_or
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.logical_or"}
   (apply base/call-function @logical_or-fnptr* args)))

(defonce ^:private logical_xor-fnptr* (delay (base/name->global-function "topi.logical_xor")))
(defn logical_xor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.logical_xor"}
   (apply base/call-function @logical_xor-fnptr* args)))

(defonce ^:private matmul-fnptr* (delay (base/name->global-function "topi.matmul")))
(defn matmul
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.matmul"}
   (apply base/call-function @matmul-fnptr* args)))

(defonce ^:private matrix_set_diag-fnptr* (delay (base/name->global-function "topi.matrix_set_diag")))
(defn matrix_set_diag
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.matrix_set_diag"}
   (apply base/call-function @matrix_set_diag-fnptr* args)))

(defonce ^:private max-fnptr* (delay (base/name->global-function "topi.max")))
(defn max
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.max"}
   (apply base/call-function @max-fnptr* args)))

(defonce ^:private maximum-fnptr* (delay (base/name->global-function "topi.maximum")))
(defn maximum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.maximum"}
   (apply base/call-function @maximum-fnptr* args)))

(defonce ^:private meshgrid-fnptr* (delay (base/name->global-function "topi.meshgrid")))
(defn meshgrid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.meshgrid"}
   (apply base/call-function @meshgrid-fnptr* args)))

(defonce ^:private min-fnptr* (delay (base/name->global-function "topi.min")))
(defn min
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.min"}
   (apply base/call-function @min-fnptr* args)))

(defonce ^:private minimum-fnptr* (delay (base/name->global-function "topi.minimum")))
(defn minimum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.minimum"}
   (apply base/call-function @minimum-fnptr* args)))

(defonce ^:private mod-fnptr* (delay (base/name->global-function "topi.mod")))
(defn mod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.mod"}
   (apply base/call-function @mod-fnptr* args)))

(defonce ^:private multiply-fnptr* (delay (base/name->global-function "topi.multiply")))
(defn multiply
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.multiply"}
   (apply base/call-function @multiply-fnptr* args)))

(defonce ^:private ndarray_size-fnptr* (delay (base/name->global-function "topi.ndarray_size")))
(defn ndarray_size
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.ndarray_size"}
   (apply base/call-function @ndarray_size-fnptr* args)))

(defonce ^:private negative-fnptr* (delay (base/name->global-function "topi.negative")))
(defn negative
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.negative"}
   (apply base/call-function @negative-fnptr* args)))

(defonce ^:private not_equal-fnptr* (delay (base/name->global-function "topi.not_equal")))
(defn not_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.not_equal"}
   (apply base/call-function @not_equal-fnptr* args)))

(defonce ^:private one_hot-fnptr* (delay (base/name->global-function "topi.one_hot")))
(defn one_hot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.one_hot"}
   (apply base/call-function @one_hot-fnptr* args)))

(defonce ^:private power-fnptr* (delay (base/name->global-function "topi.power")))
(defn power
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.power"}
   (apply base/call-function @power-fnptr* args)))

(defonce ^:private prod-fnptr* (delay (base/name->global-function "topi.prod")))
(defn prod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.prod"}
   (apply base/call-function @prod-fnptr* args)))

(defonce ^:private reinterpret-fnptr* (delay (base/name->global-function "topi.reinterpret")))
(defn reinterpret
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.reinterpret"}
   (apply base/call-function @reinterpret-fnptr* args)))

(defonce ^:private repeat-fnptr* (delay (base/name->global-function "topi.repeat")))
(defn repeat
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.repeat"}
   (apply base/call-function @repeat-fnptr* args)))

(defonce ^:private reshape-fnptr* (delay (base/name->global-function "topi.reshape")))
(defn reshape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.reshape"}
   (apply base/call-function @reshape-fnptr* args)))

(defonce ^:private reverse_sequence-fnptr* (delay (base/name->global-function "topi.reverse_sequence")))
(defn reverse_sequence
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.reverse_sequence"}
   (apply base/call-function @reverse_sequence-fnptr* args)))

(defonce ^:private right_shift-fnptr* (delay (base/name->global-function "topi.right_shift")))
(defn right_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.right_shift"}
   (apply base/call-function @right_shift-fnptr* args)))

(defonce ^:private rsqrt-fnptr* (delay (base/name->global-function "topi.rsqrt")))
(defn rsqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.rsqrt"}
   (apply base/call-function @rsqrt-fnptr* args)))

(defonce ^:private sequence_mask-fnptr* (delay (base/name->global-function "topi.sequence_mask")))
(defn sequence_mask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sequence_mask"}
   (apply base/call-function @sequence_mask-fnptr* args)))

(defonce ^:private shape-fnptr* (delay (base/name->global-function "topi.shape")))
(defn shape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.shape"}
   (apply base/call-function @shape-fnptr* args)))

(defonce ^:private sigmoid-fnptr* (delay (base/name->global-function "topi.sigmoid")))
(defn sigmoid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sigmoid"}
   (apply base/call-function @sigmoid-fnptr* args)))

(defonce ^:private sign-fnptr* (delay (base/name->global-function "topi.sign")))
(defn sign
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sign"}
   (apply base/call-function @sign-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "topi.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "topi.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private sparse_to_dense-fnptr* (delay (base/name->global-function "topi.sparse_to_dense")))
(defn sparse_to_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sparse_to_dense"}
   (apply base/call-function @sparse_to_dense-fnptr* args)))

(defonce ^:private split-fnptr* (delay (base/name->global-function "topi.split")))
(defn split
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.split"}
   (apply base/call-function @split-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "topi.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private squeeze-fnptr* (delay (base/name->global-function "topi.squeeze")))
(defn squeeze
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.squeeze"}
   (apply base/call-function @squeeze-fnptr* args)))

(defonce ^:private stack-fnptr* (delay (base/name->global-function "topi.stack")))
(defn stack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.stack"}
   (apply base/call-function @stack-fnptr* args)))

(defonce ^:private strided_slice-fnptr* (delay (base/name->global-function "topi.strided_slice")))
(defn strided_slice
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.strided_slice"}
   (apply base/call-function @strided_slice-fnptr* args)))

(defonce ^:private subtract-fnptr* (delay (base/name->global-function "topi.subtract")))
(defn subtract
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.subtract"}
   (apply base/call-function @subtract-fnptr* args)))

(defonce ^:private sum-fnptr* (delay (base/name->global-function "topi.sum")))
(defn sum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.sum"}
   (apply base/call-function @sum-fnptr* args)))

(defonce ^:private take-fnptr* (delay (base/name->global-function "topi.take")))
(defn take
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.take"}
   (apply base/call-function @take-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "topi.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "topi.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private tensordot-fnptr* (delay (base/name->global-function "topi.tensordot")))
(defn tensordot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.tensordot"}
   (apply base/call-function @tensordot-fnptr* args)))

(defonce ^:private tile-fnptr* (delay (base/name->global-function "topi.tile")))
(defn tile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.tile"}
   (apply base/call-function @tile-fnptr* args)))

(defonce ^:private transpose-fnptr* (delay (base/name->global-function "topi.transpose")))
(defn transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.transpose"}
   (apply base/call-function @transpose-fnptr* args)))

(defonce ^:private unravel_index-fnptr* (delay (base/name->global-function "topi.unravel_index")))
(defn unravel_index
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.unravel_index"}
   (apply base/call-function @unravel_index-fnptr* args)))

(defonce ^:private where-fnptr* (delay (base/name->global-function "topi.where")))
(defn where
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "topi.where"}
   (apply base/call-function @where-fnptr* args)))
