(ns tvm-clj.impl.fns.relay.op._make
  (:require [tvm-clj.impl.base :as base]))

(defonce ^:private OpStrategy-fnptr* (delay (base/name->global-function "relay.op._make.OpStrategy")))
(defn OpStrategy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.OpStrategy"}
   (apply base/call-function @OpStrategy-fnptr* args)))

(defonce ^:private _variance-fnptr* (delay (base/name->global-function "relay.op._make._variance")))
(defn _variance
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make._variance"}
   (apply base/call-function @_variance-fnptr* args)))

(defonce ^:private abs-fnptr* (delay (base/name->global-function "relay.op._make.abs")))
(defn abs
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.abs"}
   (apply base/call-function @abs-fnptr* args)))

(defonce ^:private acos-fnptr* (delay (base/name->global-function "relay.op._make.acos")))
(defn acos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.acos"}
   (apply base/call-function @acos-fnptr* args)))

(defonce ^:private acosh-fnptr* (delay (base/name->global-function "relay.op._make.acosh")))
(defn acosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.acosh"}
   (apply base/call-function @acosh-fnptr* args)))

(defonce ^:private add-fnptr* (delay (base/name->global-function "relay.op._make.add")))
(defn add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.add"}
   (apply base/call-function @add-fnptr* args)))

(defonce ^:private adv_index-fnptr* (delay (base/name->global-function "relay.op._make.adv_index")))
(defn adv_index
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.adv_index"}
   (apply base/call-function @adv_index-fnptr* args)))

(defonce ^:private all-fnptr* (delay (base/name->global-function "relay.op._make.all")))
(defn all
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.all"}
   (apply base/call-function @all-fnptr* args)))

(defonce ^:private any-fnptr* (delay (base/name->global-function "relay.op._make.any")))
(defn any
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.any"}
   (apply base/call-function @any-fnptr* args)))

(defonce ^:private arange-fnptr* (delay (base/name->global-function "relay.op._make.arange")))
(defn arange
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.arange"}
   (apply base/call-function @arange-fnptr* args)))

(defonce ^:private argmax-fnptr* (delay (base/name->global-function "relay.op._make.argmax")))
(defn argmax
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.argmax"}
   (apply base/call-function @argmax-fnptr* args)))

(defonce ^:private argmin-fnptr* (delay (base/name->global-function "relay.op._make.argmin")))
(defn argmin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.argmin"}
   (apply base/call-function @argmin-fnptr* args)))

(defonce ^:private argsort-fnptr* (delay (base/name->global-function "relay.op._make.argsort")))
(defn argsort
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.argsort"}
   (apply base/call-function @argsort-fnptr* args)))

(defonce ^:private argwhere-fnptr* (delay (base/name->global-function "relay.op._make.argwhere")))
(defn argwhere
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.argwhere"}
   (apply base/call-function @argwhere-fnptr* args)))

(defonce ^:private asin-fnptr* (delay (base/name->global-function "relay.op._make.asin")))
(defn asin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.asin"}
   (apply base/call-function @asin-fnptr* args)))

(defonce ^:private asinh-fnptr* (delay (base/name->global-function "relay.op._make.asinh")))
(defn asinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.asinh"}
   (apply base/call-function @asinh-fnptr* args)))

(defonce ^:private atan-fnptr* (delay (base/name->global-function "relay.op._make.atan")))
(defn atan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.atan"}
   (apply base/call-function @atan-fnptr* args)))

(defonce ^:private atanh-fnptr* (delay (base/name->global-function "relay.op._make.atanh")))
(defn atanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.atanh"}
   (apply base/call-function @atanh-fnptr* args)))

(defonce ^:private bitwise_and-fnptr* (delay (base/name->global-function "relay.op._make.bitwise_and")))
(defn bitwise_and
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.bitwise_and"}
   (apply base/call-function @bitwise_and-fnptr* args)))

(defonce ^:private bitwise_not-fnptr* (delay (base/name->global-function "relay.op._make.bitwise_not")))
(defn bitwise_not
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.bitwise_not"}
   (apply base/call-function @bitwise_not-fnptr* args)))

(defonce ^:private bitwise_or-fnptr* (delay (base/name->global-function "relay.op._make.bitwise_or")))
(defn bitwise_or
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.bitwise_or"}
   (apply base/call-function @bitwise_or-fnptr* args)))

(defonce ^:private bitwise_xor-fnptr* (delay (base/name->global-function "relay.op._make.bitwise_xor")))
(defn bitwise_xor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.bitwise_xor"}
   (apply base/call-function @bitwise_xor-fnptr* args)))

(defonce ^:private broadcast_to-fnptr* (delay (base/name->global-function "relay.op._make.broadcast_to")))
(defn broadcast_to
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.broadcast_to"}
   (apply base/call-function @broadcast_to-fnptr* args)))

(defonce ^:private broadcast_to_like-fnptr* (delay (base/name->global-function "relay.op._make.broadcast_to_like")))
(defn broadcast_to_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.broadcast_to_like"}
   (apply base/call-function @broadcast_to_like-fnptr* args)))

(defonce ^:private ceil-fnptr* (delay (base/name->global-function "relay.op._make.ceil")))
(defn ceil
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.ceil"}
   (apply base/call-function @ceil-fnptr* args)))

(defonce ^:private clip-fnptr* (delay (base/name->global-function "relay.op._make.clip")))
(defn clip
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.clip"}
   (apply base/call-function @clip-fnptr* args)))

(defonce ^:private collapse_sum_like-fnptr* (delay (base/name->global-function "relay.op._make.collapse_sum_like")))
(defn collapse_sum_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.collapse_sum_like"}
   (apply base/call-function @collapse_sum_like-fnptr* args)))

(defonce ^:private collapse_sum_to-fnptr* (delay (base/name->global-function "relay.op._make.collapse_sum_to")))
(defn collapse_sum_to
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.collapse_sum_to"}
   (apply base/call-function @collapse_sum_to-fnptr* args)))

(defonce ^:private concatenate-fnptr* (delay (base/name->global-function "relay.op._make.concatenate")))
(defn concatenate
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.concatenate"}
   (apply base/call-function @concatenate-fnptr* args)))

(defonce ^:private contrib_reverse_reshape-fnptr* (delay (base/name->global-function "relay.op._make.contrib_reverse_reshape")))
(defn contrib_reverse_reshape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.contrib_reverse_reshape"}
   (apply base/call-function @contrib_reverse_reshape-fnptr* args)))

(defonce ^:private copy-fnptr* (delay (base/name->global-function "relay.op._make.copy")))
(defn copy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.copy"}
   (apply base/call-function @copy-fnptr* args)))

(defonce ^:private cos-fnptr* (delay (base/name->global-function "relay.op._make.cos")))
(defn cos
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.cos"}
   (apply base/call-function @cos-fnptr* args)))

(defonce ^:private cosh-fnptr* (delay (base/name->global-function "relay.op._make.cosh")))
(defn cosh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.cosh"}
   (apply base/call-function @cosh-fnptr* args)))

(defonce ^:private debug-fnptr* (delay (base/name->global-function "relay.op._make.debug")))
(defn debug
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.debug"}
   (apply base/call-function @debug-fnptr* args)))

(defonce ^:private device_copy-fnptr* (delay (base/name->global-function "relay.op._make.device_copy")))
(defn device_copy
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.device_copy"}
   (apply base/call-function @device_copy-fnptr* args)))

(defonce ^:private divide-fnptr* (delay (base/name->global-function "relay.op._make.divide")))
(defn divide
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.divide"}
   (apply base/call-function @divide-fnptr* args)))

(defonce ^:private equal-fnptr* (delay (base/name->global-function "relay.op._make.equal")))
(defn equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.equal"}
   (apply base/call-function @equal-fnptr* args)))

(defonce ^:private erf-fnptr* (delay (base/name->global-function "relay.op._make.erf")))
(defn erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.erf"}
   (apply base/call-function @erf-fnptr* args)))

(defonce ^:private exp-fnptr* (delay (base/name->global-function "relay.op._make.exp")))
(defn exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.exp"}
   (apply base/call-function @exp-fnptr* args)))

(defonce ^:private expand_dims-fnptr* (delay (base/name->global-function "relay.op._make.expand_dims")))
(defn expand_dims
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.expand_dims"}
   (apply base/call-function @expand_dims-fnptr* args)))

(defonce ^:private fast_erf-fnptr* (delay (base/name->global-function "relay.op._make.fast_erf")))
(defn fast_erf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.fast_erf"}
   (apply base/call-function @fast_erf-fnptr* args)))

(defonce ^:private fast_exp-fnptr* (delay (base/name->global-function "relay.op._make.fast_exp")))
(defn fast_exp
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.fast_exp"}
   (apply base/call-function @fast_exp-fnptr* args)))

(defonce ^:private fast_tanh-fnptr* (delay (base/name->global-function "relay.op._make.fast_tanh")))
(defn fast_tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.fast_tanh"}
   (apply base/call-function @fast_tanh-fnptr* args)))

(defonce ^:private fixed_point_multiply-fnptr* (delay (base/name->global-function "relay.op._make.fixed_point_multiply")))
(defn fixed_point_multiply
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.fixed_point_multiply"}
   (apply base/call-function @fixed_point_multiply-fnptr* args)))

(defonce ^:private floor-fnptr* (delay (base/name->global-function "relay.op._make.floor")))
(defn floor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.floor"}
   (apply base/call-function @floor-fnptr* args)))

(defonce ^:private floor_divide-fnptr* (delay (base/name->global-function "relay.op._make.floor_divide")))
(defn floor_divide
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.floor_divide"}
   (apply base/call-function @floor_divide-fnptr* args)))

(defonce ^:private floor_mod-fnptr* (delay (base/name->global-function "relay.op._make.floor_mod")))
(defn floor_mod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.floor_mod"}
   (apply base/call-function @floor_mod-fnptr* args)))

(defonce ^:private full-fnptr* (delay (base/name->global-function "relay.op._make.full")))
(defn full
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.full"}
   (apply base/call-function @full-fnptr* args)))

(defonce ^:private full_like-fnptr* (delay (base/name->global-function "relay.op._make.full_like")))
(defn full_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.full_like"}
   (apply base/call-function @full_like-fnptr* args)))

(defonce ^:private gather-fnptr* (delay (base/name->global-function "relay.op._make.gather")))
(defn gather
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.gather"}
   (apply base/call-function @gather-fnptr* args)))

(defonce ^:private gather_nd-fnptr* (delay (base/name->global-function "relay.op._make.gather_nd")))
(defn gather_nd
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.gather_nd"}
   (apply base/call-function @gather_nd-fnptr* args)))

(defonce ^:private greater-fnptr* (delay (base/name->global-function "relay.op._make.greater")))
(defn greater
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.greater"}
   (apply base/call-function @greater-fnptr* args)))

(defonce ^:private greater_equal-fnptr* (delay (base/name->global-function "relay.op._make.greater_equal")))
(defn greater_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.greater_equal"}
   (apply base/call-function @greater_equal-fnptr* args)))

(defonce ^:private isfinite-fnptr* (delay (base/name->global-function "relay.op._make.isfinite")))
(defn isfinite
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.isfinite"}
   (apply base/call-function @isfinite-fnptr* args)))

(defonce ^:private isinf-fnptr* (delay (base/name->global-function "relay.op._make.isinf")))
(defn isinf
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.isinf"}
   (apply base/call-function @isinf-fnptr* args)))

(defonce ^:private isnan-fnptr* (delay (base/name->global-function "relay.op._make.isnan")))
(defn isnan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.isnan"}
   (apply base/call-function @isnan-fnptr* args)))

(defonce ^:private layout_transform-fnptr* (delay (base/name->global-function "relay.op._make.layout_transform")))
(defn layout_transform
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.layout_transform"}
   (apply base/call-function @layout_transform-fnptr* args)))

(defonce ^:private left_shift-fnptr* (delay (base/name->global-function "relay.op._make.left_shift")))
(defn left_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.left_shift"}
   (apply base/call-function @left_shift-fnptr* args)))

(defonce ^:private less-fnptr* (delay (base/name->global-function "relay.op._make.less")))
(defn less
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.less"}
   (apply base/call-function @less-fnptr* args)))

(defonce ^:private less_equal-fnptr* (delay (base/name->global-function "relay.op._make.less_equal")))
(defn less_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.less_equal"}
   (apply base/call-function @less_equal-fnptr* args)))

(defonce ^:private log-fnptr* (delay (base/name->global-function "relay.op._make.log")))
(defn log
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.log"}
   (apply base/call-function @log-fnptr* args)))

(defonce ^:private log10-fnptr* (delay (base/name->global-function "relay.op._make.log10")))
(defn log10
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.log10"}
   (apply base/call-function @log10-fnptr* args)))

(defonce ^:private log2-fnptr* (delay (base/name->global-function "relay.op._make.log2")))
(defn log2
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.log2"}
   (apply base/call-function @log2-fnptr* args)))

(defonce ^:private logical_and-fnptr* (delay (base/name->global-function "relay.op._make.logical_and")))
(defn logical_and
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.logical_and"}
   (apply base/call-function @logical_and-fnptr* args)))

(defonce ^:private logical_not-fnptr* (delay (base/name->global-function "relay.op._make.logical_not")))
(defn logical_not
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.logical_not"}
   (apply base/call-function @logical_not-fnptr* args)))

(defonce ^:private logical_or-fnptr* (delay (base/name->global-function "relay.op._make.logical_or")))
(defn logical_or
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.logical_or"}
   (apply base/call-function @logical_or-fnptr* args)))

(defonce ^:private logical_xor-fnptr* (delay (base/name->global-function "relay.op._make.logical_xor")))
(defn logical_xor
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.logical_xor"}
   (apply base/call-function @logical_xor-fnptr* args)))

(defonce ^:private matrix_set_diag-fnptr* (delay (base/name->global-function "relay.op._make.matrix_set_diag")))
(defn matrix_set_diag
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.matrix_set_diag"}
   (apply base/call-function @matrix_set_diag-fnptr* args)))

(defonce ^:private max-fnptr* (delay (base/name->global-function "relay.op._make.max")))
(defn max
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.max"}
   (apply base/call-function @max-fnptr* args)))

(defonce ^:private maximum-fnptr* (delay (base/name->global-function "relay.op._make.maximum")))
(defn maximum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.maximum"}
   (apply base/call-function @maximum-fnptr* args)))

(defonce ^:private mean-fnptr* (delay (base/name->global-function "relay.op._make.mean")))
(defn mean
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.mean"}
   (apply base/call-function @mean-fnptr* args)))

(defonce ^:private meshgrid-fnptr* (delay (base/name->global-function "relay.op._make.meshgrid")))
(defn meshgrid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.meshgrid"}
   (apply base/call-function @meshgrid-fnptr* args)))

(defonce ^:private min-fnptr* (delay (base/name->global-function "relay.op._make.min")))
(defn min
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.min"}
   (apply base/call-function @min-fnptr* args)))

(defonce ^:private minimum-fnptr* (delay (base/name->global-function "relay.op._make.minimum")))
(defn minimum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.minimum"}
   (apply base/call-function @minimum-fnptr* args)))

(defonce ^:private mod-fnptr* (delay (base/name->global-function "relay.op._make.mod")))
(defn mod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.mod"}
   (apply base/call-function @mod-fnptr* args)))

(defonce ^:private multiply-fnptr* (delay (base/name->global-function "relay.op._make.multiply")))
(defn multiply
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.multiply"}
   (apply base/call-function @multiply-fnptr* args)))

(defonce ^:private ndarray_size-fnptr* (delay (base/name->global-function "relay.op._make.ndarray_size")))
(defn ndarray_size
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.ndarray_size"}
   (apply base/call-function @ndarray_size-fnptr* args)))

(defonce ^:private negative-fnptr* (delay (base/name->global-function "relay.op._make.negative")))
(defn negative
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.negative"}
   (apply base/call-function @negative-fnptr* args)))

(defonce ^:private not_equal-fnptr* (delay (base/name->global-function "relay.op._make.not_equal")))
(defn not_equal
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.not_equal"}
   (apply base/call-function @not_equal-fnptr* args)))

(defonce ^:private one_hot-fnptr* (delay (base/name->global-function "relay.op._make.one_hot")))
(defn one_hot
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.one_hot"}
   (apply base/call-function @one_hot-fnptr* args)))

(defonce ^:private ones-fnptr* (delay (base/name->global-function "relay.op._make.ones")))
(defn ones
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.ones"}
   (apply base/call-function @ones-fnptr* args)))

(defonce ^:private ones_like-fnptr* (delay (base/name->global-function "relay.op._make.ones_like")))
(defn ones_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.ones_like"}
   (apply base/call-function @ones_like-fnptr* args)))

(defonce ^:private power-fnptr* (delay (base/name->global-function "relay.op._make.power")))
(defn power
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.power"}
   (apply base/call-function @power-fnptr* args)))

(defonce ^:private prod-fnptr* (delay (base/name->global-function "relay.op._make.prod")))
(defn prod
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.prod"}
   (apply base/call-function @prod-fnptr* args)))

(defonce ^:private repeat-fnptr* (delay (base/name->global-function "relay.op._make.repeat")))
(defn repeat
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.repeat"}
   (apply base/call-function @repeat-fnptr* args)))

(defonce ^:private reshape-fnptr* (delay (base/name->global-function "relay.op._make.reshape")))
(defn reshape
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.reshape"}
   (apply base/call-function @reshape-fnptr* args)))

(defonce ^:private reshape_like-fnptr* (delay (base/name->global-function "relay.op._make.reshape_like")))
(defn reshape_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.reshape_like"}
   (apply base/call-function @reshape_like-fnptr* args)))

(defonce ^:private reverse-fnptr* (delay (base/name->global-function "relay.op._make.reverse")))
(defn reverse
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.reverse"}
   (apply base/call-function @reverse-fnptr* args)))

(defonce ^:private reverse_sequence-fnptr* (delay (base/name->global-function "relay.op._make.reverse_sequence")))
(defn reverse_sequence
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.reverse_sequence"}
   (apply base/call-function @reverse_sequence-fnptr* args)))

(defonce ^:private right_shift-fnptr* (delay (base/name->global-function "relay.op._make.right_shift")))
(defn right_shift
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.right_shift"}
   (apply base/call-function @right_shift-fnptr* args)))

(defonce ^:private round-fnptr* (delay (base/name->global-function "relay.op._make.round")))
(defn round
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.round"}
   (apply base/call-function @round-fnptr* args)))

(defonce ^:private rsqrt-fnptr* (delay (base/name->global-function "relay.op._make.rsqrt")))
(defn rsqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.rsqrt"}
   (apply base/call-function @rsqrt-fnptr* args)))

(defonce ^:private scatter-fnptr* (delay (base/name->global-function "relay.op._make.scatter")))
(defn scatter
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.scatter"}
   (apply base/call-function @scatter-fnptr* args)))

(defonce ^:private scatter_add-fnptr* (delay (base/name->global-function "relay.op._make.scatter_add")))
(defn scatter_add
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.scatter_add"}
   (apply base/call-function @scatter_add-fnptr* args)))

(defonce ^:private sequence_mask-fnptr* (delay (base/name->global-function "relay.op._make.sequence_mask")))
(defn sequence_mask
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sequence_mask"}
   (apply base/call-function @sequence_mask-fnptr* args)))

(defonce ^:private shape_of-fnptr* (delay (base/name->global-function "relay.op._make.shape_of")))
(defn shape_of
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.shape_of"}
   (apply base/call-function @shape_of-fnptr* args)))

(defonce ^:private sigmoid-fnptr* (delay (base/name->global-function "relay.op._make.sigmoid")))
(defn sigmoid
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sigmoid"}
   (apply base/call-function @sigmoid-fnptr* args)))

(defonce ^:private sign-fnptr* (delay (base/name->global-function "relay.op._make.sign")))
(defn sign
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sign"}
   (apply base/call-function @sign-fnptr* args)))

(defonce ^:private sin-fnptr* (delay (base/name->global-function "relay.op._make.sin")))
(defn sin
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sin"}
   (apply base/call-function @sin-fnptr* args)))

(defonce ^:private sinh-fnptr* (delay (base/name->global-function "relay.op._make.sinh")))
(defn sinh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sinh"}
   (apply base/call-function @sinh-fnptr* args)))

(defonce ^:private slice_like-fnptr* (delay (base/name->global-function "relay.op._make.slice_like")))
(defn slice_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.slice_like"}
   (apply base/call-function @slice_like-fnptr* args)))

(defonce ^:private sparse_to_dense-fnptr* (delay (base/name->global-function "relay.op._make.sparse_to_dense")))
(defn sparse_to_dense
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sparse_to_dense"}
   (apply base/call-function @sparse_to_dense-fnptr* args)))

(defonce ^:private split-fnptr* (delay (base/name->global-function "relay.op._make.split")))
(defn split
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.split"}
   (apply base/call-function @split-fnptr* args)))

(defonce ^:private sqrt-fnptr* (delay (base/name->global-function "relay.op._make.sqrt")))
(defn sqrt
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sqrt"}
   (apply base/call-function @sqrt-fnptr* args)))

(defonce ^:private squeeze-fnptr* (delay (base/name->global-function "relay.op._make.squeeze")))
(defn squeeze
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.squeeze"}
   (apply base/call-function @squeeze-fnptr* args)))

(defonce ^:private stack-fnptr* (delay (base/name->global-function "relay.op._make.stack")))
(defn stack
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.stack"}
   (apply base/call-function @stack-fnptr* args)))

(defonce ^:private strided_set-fnptr* (delay (base/name->global-function "relay.op._make.strided_set")))
(defn strided_set
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.strided_set"}
   (apply base/call-function @strided_set-fnptr* args)))

(defonce ^:private strided_slice-fnptr* (delay (base/name->global-function "relay.op._make.strided_slice")))
(defn strided_slice
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.strided_slice"}
   (apply base/call-function @strided_slice-fnptr* args)))

(defonce ^:private subtract-fnptr* (delay (base/name->global-function "relay.op._make.subtract")))
(defn subtract
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.subtract"}
   (apply base/call-function @subtract-fnptr* args)))

(defonce ^:private sum-fnptr* (delay (base/name->global-function "relay.op._make.sum")))
(defn sum
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.sum"}
   (apply base/call-function @sum-fnptr* args)))

(defonce ^:private take-fnptr* (delay (base/name->global-function "relay.op._make.take")))
(defn take
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.take"}
   (apply base/call-function @take-fnptr* args)))

(defonce ^:private tan-fnptr* (delay (base/name->global-function "relay.op._make.tan")))
(defn tan
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.tan"}
   (apply base/call-function @tan-fnptr* args)))

(defonce ^:private tanh-fnptr* (delay (base/name->global-function "relay.op._make.tanh")))
(defn tanh
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.tanh"}
   (apply base/call-function @tanh-fnptr* args)))

(defonce ^:private tile-fnptr* (delay (base/name->global-function "relay.op._make.tile")))
(defn tile
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.tile"}
   (apply base/call-function @tile-fnptr* args)))

(defonce ^:private topk-fnptr* (delay (base/name->global-function "relay.op._make.topk")))
(defn topk
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.topk"}
   (apply base/call-function @topk-fnptr* args)))

(defonce ^:private transpose-fnptr* (delay (base/name->global-function "relay.op._make.transpose")))
(defn transpose
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.transpose"}
   (apply base/call-function @transpose-fnptr* args)))

(defonce ^:private trunc-fnptr* (delay (base/name->global-function "relay.op._make.trunc")))
(defn trunc
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.trunc"}
   (apply base/call-function @trunc-fnptr* args)))

(defonce ^:private unravel_index-fnptr* (delay (base/name->global-function "relay.op._make.unravel_index")))
(defn unravel_index
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.unravel_index"}
   (apply base/call-function @unravel_index-fnptr* args)))

(defonce ^:private where-fnptr* (delay (base/name->global-function "relay.op._make.where")))
(defn where
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.where"}
   (apply base/call-function @where-fnptr* args)))

(defonce ^:private zeros-fnptr* (delay (base/name->global-function "relay.op._make.zeros")))
(defn zeros
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.zeros"}
   (apply base/call-function @zeros-fnptr* args)))

(defonce ^:private zeros_like-fnptr* (delay (base/name->global-function "relay.op._make.zeros_like")))
(defn zeros_like
 "TVM PackedFn"
 [& args]
 (with-bindings {#'base/fn-name "relay.op._make.zeros_like"}
   (apply base/call-function @zeros_like-fnptr* args)))

