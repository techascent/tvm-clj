(ns tvm-clj.jna.fns.topi
  (:require [tvm-clj.jna.base :as jna-base]))

(let [gfn* (delay (jna-base/name->global-function "topi.TEST_create_target"))]
  (defn TEST_create_target
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.TEST_create_target"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.acos"))]
  (defn acos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.acos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.acosh"))]
  (defn acosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.acosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.add"))]
  (defn add
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.add"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.adv_index"))]
  (defn adv_index
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.adv_index"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.all"))]
  (defn all
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.all"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.any"))]
  (defn any
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.any"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.arange"))]
  (defn arange
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.arange"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.argmax"))]
  (defn argmax
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.argmax"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.argmin"))]
  (defn argmin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.argmin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.asin"))]
  (defn asin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.asin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.asinh"))]
  (defn asinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.asinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.atan"))]
  (defn atan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.atan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.atanh"))]
  (defn atanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.atanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_and"))]
  (defn bitwise_and
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.bitwise_and"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_not"))]
  (defn bitwise_not
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.bitwise_not"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_or"))]
  (defn bitwise_or
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.bitwise_or"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.bitwise_xor"))]
  (defn bitwise_xor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.bitwise_xor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.broadcast_to"))]
  (defn broadcast_to
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.broadcast_to"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cast"))]
  (defn cast
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cast"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.clip"))]
  (defn clip
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.clip"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.concatenate"))]
  (defn concatenate
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.concatenate"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cos"))]
  (defn cos
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cos"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.cosh"))]
  (defn cosh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.cosh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.divide"))]
  (defn divide
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.divide"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.elemwise_sum"))]
  (defn elemwise_sum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.elemwise_sum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.equal"))]
  (defn equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.erf"))]
  (defn erf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.erf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.exp"))]
  (defn exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.expand_dims"))]
  (defn expand_dims
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.expand_dims"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.fast_erf"))]
  (defn fast_erf
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.fast_erf"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.fast_exp"))]
  (defn fast_exp
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.fast_exp"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.fast_tanh"))]
  (defn fast_tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.fast_tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.flip"))]
  (defn flip
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.flip"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.floor_divide"))]
  (defn floor_divide
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.floor_divide"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.floor_mod"))]
  (defn floor_mod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.floor_mod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.full"))]
  (defn full
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.full"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.full_like"))]
  (defn full_like
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.full_like"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.gather"))]
  (defn gather
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.gather"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.gather_nd"))]
  (defn gather_nd
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.gather_nd"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.greater"))]
  (defn greater
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.greater"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.greater_equal"))]
  (defn greater_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.greater_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.identity"))]
  (defn identity
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.identity"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.layout_transform"))]
  (defn layout_transform
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.layout_transform"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.left_shift"))]
  (defn left_shift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.left_shift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.less"))]
  (defn less
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.less"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.less_equal"))]
  (defn less_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.less_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.log"))]
  (defn log
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.log"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.log10"))]
  (defn log10
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.log10"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.log2"))]
  (defn log2
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.log2"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.logical_and"))]
  (defn logical_and
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.logical_and"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.logical_not"))]
  (defn logical_not
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.logical_not"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.logical_or"))]
  (defn logical_or
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.logical_or"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.logical_xor"))]
  (defn logical_xor
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.logical_xor"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.matmul"))]
  (defn matmul
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.matmul"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.matrix_set_diag"))]
  (defn matrix_set_diag
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.matrix_set_diag"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.max"))]
  (defn max
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.max"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.maximum"))]
  (defn maximum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.maximum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.meshgrid"))]
  (defn meshgrid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.meshgrid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.min"))]
  (defn min
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.min"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.minimum"))]
  (defn minimum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.minimum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.mod"))]
  (defn mod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.mod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.multiply"))]
  (defn multiply
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.multiply"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.ndarray_size"))]
  (defn ndarray_size
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.ndarray_size"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.negative"))]
  (defn negative
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.negative"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.not_equal"))]
  (defn not_equal
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.not_equal"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.one_hot"))]
  (defn one_hot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.one_hot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.power"))]
  (defn power
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.power"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.prod"))]
  (defn prod
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.prod"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.reinterpret"))]
  (defn reinterpret
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.reinterpret"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.repeat"))]
  (defn repeat
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.repeat"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.reshape"))]
  (defn reshape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.reshape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.reverse_sequence"))]
  (defn reverse_sequence
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.reverse_sequence"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.right_shift"))]
  (defn right_shift
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.right_shift"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.rsqrt"))]
  (defn rsqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.rsqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sequence_mask"))]
  (defn sequence_mask
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sequence_mask"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.shape"))]
  (defn shape
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.shape"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sigmoid"))]
  (defn sigmoid
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sigmoid"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sign"))]
  (defn sign
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sign"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sin"))]
  (defn sin
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sin"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sinh"))]
  (defn sinh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sinh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sparse_to_dense"))]
  (defn sparse_to_dense
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sparse_to_dense"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.split"))]
  (defn split
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.split"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sqrt"))]
  (defn sqrt
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sqrt"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.squeeze"))]
  (defn squeeze
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.squeeze"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.stack"))]
  (defn stack
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.stack"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.strided_slice"))]
  (defn strided_slice
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.strided_slice"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.subtract"))]
  (defn subtract
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.subtract"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.sum"))]
  (defn sum
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.sum"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.take"))]
  (defn take
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.take"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.tan"))]
  (defn tan
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.tan"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.tanh"))]
  (defn tanh
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.tanh"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.tensordot"))]
  (defn tensordot
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.tensordot"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.tile"))]
  (defn tile
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.tile"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.transpose"))]
  (defn transpose
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.transpose"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.unravel_index"))]
  (defn unravel_index
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.unravel_index"}
     (apply jna-base/call-function @gfn* args))))

(let [gfn* (delay (jna-base/name->global-function "topi.where"))]
  (defn where
   "TVM PackedFn"
   [& args]
   (with-bindings {#'jna-base/fn-name "topi.where"}
     (apply jna-base/call-function @gfn* args))))

