# tvm-clj

Clojure bindings and exploration of the [tvm](https://github.com/dmlc/tvm) library, part of the [dmlc](https://github.com/dmlc) ecosystem.

We have a primer on the theory [here](http://techascent.com/blog/high-performance-compilers.html) with a simple example you can read about on our [blog](http://techascent.com/blog/tvm-for-the-win.html) and try out the code yourself in the example [project](examples/src/box_blur.clj).  For a more discoursive introduction, checkout Daniel Compton's repl [podcast](https://www.therepl.net/episodes/13/).


[![Clojars Project](https://img.shields.io/clojars/v/tvm-clj.svg)](https://clojars.org/tvm-clj)


## Getting all the source

At top level:
```bash
git submodule update --init --recursive
```

## Building the TVM java bindings

```bash
sudo apt install make g++ cmake llvm-dev libopenblas-dev

## opencl support (nvidia-cuda includes this)
sudo apt install ocl-icd-* opencl-headers

## Cuda support
sudo apt install  nvidia-cuda-toolkit

## intel graphics adapter support
sudo apt install beignet beignet-opencl-icd


## MKL support if you choose.  I don't use it generally so this is very optional.
curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | sudo apt-key add -
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
## Find the version of mkl...I would take the latest.
apt-cache search mkl-64bit
## ...
sudo apt-get install intel-mkl-64bit-2019.5-075


makedir -p tvm/build
# Config setup for intel and such.
# Base config.cmake file only has support for opencl.  If you want
# CUDA, CUDNN, or MKL I suggest you edit the config file after you copy
# it.
cp config.cmake tvm/build/

pushd tvm/build

cmake ..
make -j8
popd
```

At this point you should have native libraries under tvm/build/

If you want the binaries packaged with the jar, run:

```clojure
lein jni
```

This will copy the libs into a platform-specific directory that jna should find.

Another options is to install the tvm libs themselves.  We recommend this pathway as
then the tvm libraries will work with the python bindings.  In fact, it can be worth it
to install the python bindings as there are a lot of examples in python that are
instructive to work with.


## More Information


* [background theoretical documentation](docs/background.md)


## License


Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
