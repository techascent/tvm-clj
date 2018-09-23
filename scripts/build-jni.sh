#!/bin/bash

rm java/tvm_clj/tvm/runtime.java

lein jni build-jni-java
lein jni build-jni
cp tvm/build/libtvm.so java/native/linux/x86_64/
cp tvm/build/libtvm_topi.so java/native/linux/x86_64
