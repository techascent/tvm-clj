# tvm-clj

Clojure bindings and exploration of the [tvm](https://github.com/dmlc/tvm) library, part of the [dmlc](https://github.com/dmlc) ecosystem.


## Justification

[tvm](https://github.com/dmlc/tvm) a system for dynamically generating high performance numeric code with backends for cpu, cuda, opencl, opengl, webassembly, vulcan, and verilog.  It has frontends mainly in python and c++ with a clear and well designed C-ABI that not only aids in the implementation of their python interface, but it also eases the binding into other language ecosystems such as the jvm and node.

tvm leverages [Halide](http://halide-lang.org).  Halide takes algorithms structured in specific ways and allows performance experimentation without affecting the output of the core algorithm.  A very solid justification for this is nicely put in these [slides](http://stellar.mit.edu/S/course/6/sp15/6.815/courseMaterial/topics/topic2/lectureNotes/14_Halide_print/14_Halide_print.pdf).  A Ph. D. was minted [here](http://people.csail.mit.edu/jrk/jrkthesis.pdf).  We also recommend watching the youtube [video](https://youtu.be/3uiEyEKji0M).


## Goals

1.  Learn about Halide and tvm and enable very clear and simple exploration of the system in clojure.  Make clojure a first class language in the dmlc ecosystem.
1.  Provide the tvm team with clear feedback and a second external implementation or a language binding on top of the C-ABI.
1.  Encourage wider adoption and exploration in terms of numerical programming; for instance a new implementation of J that carries the properties of a clojure or clojurescript ecosystem but includes all of the major concepts of J.  This would enable running some subset of J (or APL) programs (or functions) that are now far more optimized mode than before and accessible from node.js or the jvm.  It would also inform the wider discussion on numeric programming languages such as MatLab, TensorFlow, numpy, etc.
1.  Provide richer platform for binding to nnvm so that running existing networks via clojure is as seamless as possible.


## What, Concretely, Are You Talking About?


### Simple Example

tvm exposes a directed graph along with a declarative scheduling system to build high performance numerical systems for n-dimensional data.  In the example below, we dynamically create a function to add 2 vectors then compile that function for a cpu and gpu backend.  Note that the major difference between the backends lies in the scheduling; not in the algorithm itself.
[source](test/tvm_clj/api_test.clj)


### Vector Math Compiler Example

Built a small compiler that takes a statement of vector math and compiles to tvm.  This is extremely incomplete and not very efficient in terms of what is possible but
shows a vision of doing potentially entire neural network functions.

```clojure
tvm-clj.compute.compile-fn-test> (time-tests)
java tensor ops took:  "Elapsed time: 8775.659927 msecs"

hand-coded java took:  "Elapsed time: 520.54155 msecs"

Compiled (cpu) tensor took: "Elapsed time: 223.748329 msecs"

Compiled (opencl) tensor took: "Elapsed time: 64.999564 msecs"
```
[source](test/tvm_clj/compute/compile_fn_test.clj)


### Image Scaling (TVM vs OpenCV)

Faster (and correct) bilinear filtering.  Handily beats opencv::resize for bilinear in quality and code readability.


tvm (32 ms): ![tvm-results](docs/images/test.jpg)


-- Update --
-- Need to test this timing on a machine that does not throttle the cpu --
opecnv (?? ms): ![opencv-results](docs/images/ref.jpg)


* [tvm-clj source](src/tvm_clj/image/bilinear_reduce.clj)
* [opencv source](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/resize.cpp)


## Getting all the source

At top level:
```bash
git submodule update --init --recursive
```

## Building the TVM java bindings

```bash
sudo apt install make g++ cmake llvm-dev libopenblas-dev

## Cuda support
sudo apt install  nvidia-cuda-toolkit

## opencl support (nvidia-cuda includes this)
sudo apt install ocl-icd-* opencl-headers

## intel graphics adapter support
sudo apt install beignet beignet-opencl-icd

pushd tvm


mkdir build
cp cmake/make.config build
pushd build

## now edit tvm/build/config.cmake to appropriate for your system. I have
## tested openblas cuda, opencl.
cmake ..

make -j8
popd
popd

scripts/build-jni.sh
```


At this point you should have the bindings under `java/tvm_clj/tvm/runtime/java` and a couple native libraries
under the `java/native/linux/x86_64` pathway.

Building a jar or uberjar will package all of these things into a good place.


## License


Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
