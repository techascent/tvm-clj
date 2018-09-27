#!/bin/bash

## This script is used to install enough python to file bugs with the tvm
## maintainers.  Bugs should be filed in python until clojure is part of
## the accepted languages for tvm.


sudo apt install -y --no-install-recommends \
     libboost-python-dev \
     build-essential \
     python3-dev \
     python3-pip \
     python3-setuptools \
     python3-wheel


pip3 install \
     numpy \
     decorator \
     Pillow \
     scipy \
     sklearn \
     opencv-python \
     scikit-image

pushd tvm
pushd python
python3 setup.py install --user
popd
pushd topi/python
python3 setup.py install --user
popd
pushd nnvm/python
python3 setup.py install --user
popd
popd
			 
