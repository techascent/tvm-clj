## Expand knowledge of kernels, graph system.
* System only supports injective kernels.  There are scanning, pooling, reductive, at least more to understand.
* Introspection into compilation process.
* Expand auto-compilation system using knowledge in the TOPI library to support things like conv-2d an deconv-2d.
* Build some more complex graphs, potentially the yolo2 loss function or something involved.
* A far more complex system would be MTCNN.  This would involve supporting sorting along with limited device->host transfer that should be optimized should the device be CPU.


## Build nnvm bindings on top of this base
* Currently NNVM bindings for java are built on very difficult to use primitives and they don't contain the full tvm bindings; i.e. you can't do everything from java because it only has a partial binding to tvm.  These bindings are fairly complete so a layer of bindings to nnvm built with this library underneath would provide much better access to a large, highly optimized system from clojure for running neural networks.
