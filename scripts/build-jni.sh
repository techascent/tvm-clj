#!/bin/bash

rm java/tvm_clj/tvm/runtime.java

lein jni build-jni-java
lein jni build-jni
copy-libs.sh
