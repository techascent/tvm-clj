#!/bin/bash

rm java/tvm_clj/tvm/runtime.java

lein jni build-jni-java
lein jni build-jni
cp tvm/lib/libtvm.so java/native/linux/x86_64/
cp tvm/lib/libtvm_topi.so java/native/linux/x86_64
