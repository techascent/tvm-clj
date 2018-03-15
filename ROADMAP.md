## Initial system for kernel combing
* The tensors are right now designed to implement the operations discretely.  This is not, however, the most efficient way to use TVM at all.  The best by far is to combine all the kernels into one function, and in some cases to combine multiple kernels into each other.

## Implement TOPI bindings
* TOPI is the tvm library of tested functions built via TVM.  It contains nontrivial functions to write yourself and hey, they are all right there.  This requires somehow compiling and loading the topi shared library along with definitions, documentation, and bindings for the functions in the library. 
 
## Build nnvm bindings on top of this base
* Currently NNVM bindings for java are built on very difficult to use primitives and they don't contain the full tvm bindings; i.e. you can't do everything from java because it only has a partial binding to tvm.  These bindings are fairly complete so a layer of bindings to nnvm built with this library underneath would provide much better access to a large, highly optimized system from clojure for running neural networks.

## Build out compute tensor implementation
* This is mainly to have a working generic and highly optimized n-dimensional base layer that is easy to extend to other backends.  This requires changes to the tech.compute tensor interface; there are assumptions in there that cannot be efficiently compiled by this (tvm) system.  These cases should be found and systematically fixed in the mean time building up the systematic knowledge of the halide compiler.  There are also more possibilities with a tvm-backed system for example:
```clojure
(def fancy-function 
  (tvm/compilation-context
    ;;some complex equation like yolo loss
	)
```
