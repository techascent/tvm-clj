# tvm-clj

Clojure bindings and exploration of the tvm library, part of the aws NNVM ecosystem.


## Justification

tvm a system for dynamically generating high performance numeric code with backends for cpu, cuda, opencl, opengl, webassembly, vulcan, and verilog.  It has frontends mainly in python and c++ with a clear and well designed C-ABI that not only aids in the implementation of their python interface, but it also eases the binding into other language ecosystems such as the jvm and node.

The core mechanism of leverage of tvm is the [halide](halide-lang.org) library.  This is a library specifically designed to take algorithms structured in specific ways and allow performance experimentation without affecting the output of the core algorithm.  A very solid justification for this is nicely put in these [slides](http://stellar.mit.edu/S/course/6/sp15/6.815/courseMaterial/topics/topic2/lectureNotes/14_Halide_print/14_Halide_print.pdf).  A Ph. D. was minted [here](http://people.csail.mit.edu/jrk/jrkthesis.pdf).  We also recommend watching the youtube [video](https://youtu.be/3uiEyEKji0M).


## The Goals 

1.  Learn about Halide and tvm and enable very clear and simple exploration of the system in clojure.
1.  Provide the tvm team with clear feedback and a second external implementation or a language binding on top of the C-ABI.
1.  Leverage lessons learned to enable a simple clojurescript binding to node.js ideally providing identical API and abstraction layer to the javascript ecosystem.
1.  Encourage wider adoption and exploration in terms of numerical programming; for instance a new implementation of J that carries the properties of a clojure or clojurescript ecosystem but includes all of the concepts.  This would enable running some subset of J (or APL) programs (or functions) that were potentially far more optimized mode than before and accessible from node.js or the jvm.  It would also inform the wider discussion on numeric programming languages such as MatLab, TensorFlow, numpy, etc.


## Getting all the source

At top level:
```bash
git submodule update --init --recursive
```

## Building the TVM java bindings

```bash
sudo apt install cmake maven llvm llvm-4.0-dev libblas-dev
pushd tvm

## now edit make/config.mk to appropriate for your system; I built cuda and opencl with cublas support
## but without ROC support.  I also added in LLVM (required) and blas support.
make -j8
popd

scripts/build-jni.sh
```


At this point you should have the bindings under `java/tvm_clj/tvm/runtime/java` and a couple native libraries
under the `java/native/linux/x86_64` pathway.

Building a jar or uberjar will package all of these things into a good place.


## License


Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
