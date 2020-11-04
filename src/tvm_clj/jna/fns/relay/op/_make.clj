(ns tvm-clj.jna.fns.relay.op._make
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.OpStrategy"))]
  (defn OpStrategy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.OpStrategy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make._variance"))]
  (defn _variance
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make._variance"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.abs"))]
  (defn abs
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.abs"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.acos"))]
  (defn acos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.acos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.acosh"))]
  (defn acosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.acosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.add"))]
  (defn add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.adv_index"))]
  (defn adv_index
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.adv_index"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.all"))]
  (defn all
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.all"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.any"))]
  (defn any
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.any"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.arange"))]
  (defn arange
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.arange"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argmax"))]
  (defn argmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.argmax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argmin"))]
  (defn argmin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.argmin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argsort"))]
  (defn argsort
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.argsort"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.argwhere"))]
  (defn argwhere
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.argwhere"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.asin"))]
  (defn asin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.asin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.asinh"))]
  (defn asinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.asinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.atan"))]
  (defn atan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.atan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.atanh"))]
  (defn atanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.atanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_and"))]
  (defn bitwise_and
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.bitwise_and"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_not"))]
  (defn bitwise_not
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.bitwise_not"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_or"))]
  (defn bitwise_or
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.bitwise_or"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.bitwise_xor"))]
  (defn bitwise_xor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.bitwise_xor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.broadcast_to"))]
  (defn broadcast_to
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.broadcast_to"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.broadcast_to_like"))]
  (defn broadcast_to_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.broadcast_to_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ceil"))]
  (defn ceil
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.ceil"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.clip"))]
  (defn clip
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.clip"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.collapse_sum_like"))]
  (defn collapse_sum_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.collapse_sum_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.collapse_sum_to"))]
  (defn collapse_sum_to
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.collapse_sum_to"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.concatenate"))]
  (defn concatenate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.concatenate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.contrib_reverse_reshape"))]
  (defn contrib_reverse_reshape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.contrib_reverse_reshape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.copy"))]
  (defn copy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.copy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.cos"))]
  (defn cos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.cos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.cosh"))]
  (defn cosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.cosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.debug"))]
  (defn debug
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.debug"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.device_copy"))]
  (defn device_copy
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.device_copy"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.divide"))]
  (defn divide
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.divide"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.equal"))]
  (defn equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.erf"))]
  (defn erf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.erf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.expand_dims"))]
  (defn expand_dims
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.expand_dims"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fast_erf"))]
  (defn fast_erf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.fast_erf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fast_exp"))]
  (defn fast_exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.fast_exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fast_tanh"))]
  (defn fast_tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.fast_tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.fixed_point_multiply"))]
  (defn fixed_point_multiply
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.fixed_point_multiply"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.floor"))]
  (defn floor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.floor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.floor_divide"))]
  (defn floor_divide
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.floor_divide"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.floor_mod"))]
  (defn floor_mod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.floor_mod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.full"))]
  (defn full
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.full"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.full_like"))]
  (defn full_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.full_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.gather"))]
  (defn gather
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.gather"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.gather_nd"))]
  (defn gather_nd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.gather_nd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.greater"))]
  (defn greater
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.greater"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.greater_equal"))]
  (defn greater_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.greater_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.isfinite"))]
  (defn isfinite
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.isfinite"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.isinf"))]
  (defn isinf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.isinf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.isnan"))]
  (defn isnan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.isnan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.layout_transform"))]
  (defn layout_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.layout_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.left_shift"))]
  (defn left_shift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.left_shift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.less"))]
  (defn less
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.less"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.less_equal"))]
  (defn less_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.less_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.log10"))]
  (defn log10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.log10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.log2"))]
  (defn log2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.log2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_and"))]
  (defn logical_and
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.logical_and"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_not"))]
  (defn logical_not
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.logical_not"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_or"))]
  (defn logical_or
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.logical_or"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.logical_xor"))]
  (defn logical_xor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.logical_xor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.matrix_set_diag"))]
  (defn matrix_set_diag
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.matrix_set_diag"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.max"))]
  (defn max
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.max"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.maximum"))]
  (defn maximum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.maximum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.mean"))]
  (defn mean
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.mean"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.meshgrid"))]
  (defn meshgrid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.meshgrid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.min"))]
  (defn min
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.min"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.minimum"))]
  (defn minimum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.minimum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.mod"))]
  (defn mod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.mod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.multiply"))]
  (defn multiply
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.multiply"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ndarray_size"))]
  (defn ndarray_size
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.ndarray_size"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.negative"))]
  (defn negative
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.negative"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.not_equal"))]
  (defn not_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.not_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.one_hot"))]
  (defn one_hot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.one_hot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ones"))]
  (defn ones
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.ones"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.ones_like"))]
  (defn ones_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.ones_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.power"))]
  (defn power
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.power"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.prod"))]
  (defn prod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.prod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.repeat"))]
  (defn repeat
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.repeat"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reshape"))]
  (defn reshape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.reshape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reshape_like"))]
  (defn reshape_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.reshape_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reverse"))]
  (defn reverse
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.reverse"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.reverse_sequence"))]
  (defn reverse_sequence
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.reverse_sequence"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.right_shift"))]
  (defn right_shift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.right_shift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.round"))]
  (defn round
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.round"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.rsqrt"))]
  (defn rsqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.rsqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.scatter"))]
  (defn scatter
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.scatter"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.scatter_add"))]
  (defn scatter_add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.scatter_add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sequence_mask"))]
  (defn sequence_mask
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sequence_mask"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.shape_of"))]
  (defn shape_of
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.shape_of"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sigmoid"))]
  (defn sigmoid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sigmoid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sign"))]
  (defn sign
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sign"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sin"))]
  (defn sin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sinh"))]
  (defn sinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.slice_like"))]
  (defn slice_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.slice_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sparse_to_dense"))]
  (defn sparse_to_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sparse_to_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.split"))]
  (defn split
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.split"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.squeeze"))]
  (defn squeeze
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.squeeze"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.stack"))]
  (defn stack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.stack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.strided_set"))]
  (defn strided_set
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.strided_set"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.strided_slice"))]
  (defn strided_slice
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.strided_slice"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.subtract"))]
  (defn subtract
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.subtract"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.sum"))]
  (defn sum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.sum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.take"))]
  (defn take
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.take"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.tan"))]
  (defn tan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.tan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.tile"))]
  (defn tile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.tile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.topk"))]
  (defn topk
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.topk"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.transpose"))]
  (defn transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.trunc"))]
  (defn trunc
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.trunc"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.unravel_index"))]
  (defn unravel_index
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.unravel_index"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.where"))]
  (defn where
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.where"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.zeros"))]
  (defn zeros
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.zeros"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "relay.op._make.zeros_like"))]
  (defn zeros_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "relay.op._make.zeros_like"}
     (apply jna-base/call-function @gfn* args))))

