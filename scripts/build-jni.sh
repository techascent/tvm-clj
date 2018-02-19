#!/bin/bash

rm java/tvm_clj/tvm/runtime.java

lein run build-jni-java
lein run build-jni
cp tvm/lib/libtvm.so java/native/linux/x86_64/
