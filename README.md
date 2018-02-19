# tvm-clj

Clojure bindings and exploration of the tvm library, part of the aws NNVM ecosystem.


## Getting all the source

At top level:
```bash
git submodule init
git submodule update

pushd tvm
git submodule init
git submodule update
popd
```

## Building the TVM java bindings

```bash
sudo apt install cmake maven llvm llvm-4.0-dev libblas-dev
pushd tvm

## now edit make/config.mk to appropriate for your system; I built cuda and opencl with cublas support
## but without ROC support.  I also added in LLVM and blas support.
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
