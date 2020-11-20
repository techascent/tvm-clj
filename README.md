# tvm-clj

Clojure bindings and exploration of the [tvm](https://github.com/apache/incubator-tvm) library.

TVM is a high performance compiler for ND numeric code.  In it's simplest form, it works via
4 steps: 

1.  Define an AST.
2.  Schedule the AST, doing things such as tiling and operation or caching a partial
    result in GPU shared memory.  This allows us to make transformations to the algorithm which
        allow us to map the algorithm to specific hardware such as GPU's, FPGA's, web-based
	backends such as wasm, graphics backends such as OpenGL and Vulkan and low powered IoT
	platforms such as [microcontrollers](https://tvm.apache.org/2020/06/04/tinyml-how-tvm-is-taming-tiny).
	These transformations are are guaranteed not to break the algorithm so they are very safe from
	a correctness viewpoint but also allow powerful vectorizing, SIMD, and 
	[SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) optimizations.
3.  Compile the AST to a specific hardware profile.  TVM has backends to a 
    [wide variety of hardware](https://github.com/apache/incubator-tvm/blob/main/python/tvm/_ffi/runtime_ctypes.py#L156)
	including, as mentioned, extremely optimized versions for x86 and ARM CPUs, Cuda, and OpenCL.
4.  Load your function and call it.



* [API Documents](https://techascent.github.io/tvm-clj/)
* [simple tests](test/tvm_clj/tvm_test.clj)


## Getting all the source

At top level:
```bash
git submodule update --init --recursive
```

## Building TVM

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


mkdir -p tvm/build
# Config setup for intel and such.
# Base config.cmake file only has support for opencl.  If you want
# CUDA, CUDNN, or MKL I suggest you edit the config file after you copy
# it.
cp config.cmake tvm/build/

cd tvm/build

cmake ..
make -j8


```

This will copy the libs into a platform-specific directory that jna should find.

Another options is to install the tvm libs themselves.  We recommend this pathway as
then the tvm libraries will work with the python bindings.  In fact, it can be worth it
to install the python bindings as there are a lot of examples in python that are
instructive to work with.


## More Information


* [background theoretical documentation](topics/background.md)


## License


Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
