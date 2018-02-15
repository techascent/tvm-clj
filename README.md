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
sudo apt install cmake maven llvm-4.0-dev libblas-dev
pushd tvm

## now edit config/config.mk to appropriate for your system; I built cuda and opencl with cublas support
## but without ROC support.  I also added in LLVM and blas support.
make -j8
make jvmpkg
make jvminstall
```

At this point you should have a line in your output that looks like:

```
[INFO] Installing /home/chrisn/dev/tech/tvm-clj/tvm/jvm/assembly/linux-x86_64-gpu/target/tvm4j-full-linux-x86_64-gpu-0.0.1-SNAPSHOT.jar to /home/chrisn/.m2/repository/ml/dmlc/tvm/tvm4j-full-linux-x86_64-gpu/0.0.1-SNAPSHOT/tvm4j-full-linux-x86_64-gpu-0.0.1-SNAPSHOT.jar
```

There are jvm tests included with tvm; these tests don't test anything meaningful as of the last time I checked but YYMV.


## License


Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
