# tvm-clj

Clojure bindings and exploration of the tvm library, part of the aws NNVM ecosystem.


## Justification

tvm a system for dynamically generating high performance numeric code with backends for cpu, cuda, opencl, opengl, webassembly, vulcan, and verilog.  It has frontends mainly in python and c++ with a clear and well designed C-ABI to make both the python frontend easier to write and that greatly eases binding to other language VMs, such as java or node.js.

The core mechanism of leverage of tvm is the [halide](halide-lang.org) library.  This is a library specifically designed to take algorithms structured in specific ways and allow performance experimentation without affecting the output of the core algorithm.  A very solid justification for this is nicely put in these [slides](http://stellar.mit.edu/S/course/6/sp15/6.815/courseMaterial/topics/topic2/lectureNotes/14_Halide_print/14_Halide_print.pdf).  A Ph. D. was minted [here](http://people.csail.mit.edu/jrk/jrkthesis.pdf).





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
