#!/bin/bash

set -e

IMG="tvm-dev-repl"
REPL_PORT=7000

docker build \
	   -t $IMG \
	   -f Dockerfile \
	   --build-arg USERID=$(id -u) \
	   --build-arg GROUPID=$(id -u) \
	   --build-arg USERNAME=$USER \
	   .

docker run --rm -it -u $(id -u):$(id -g) \
  -e LEIN_REPL_HOST="0.0.0.0" \
  -e LEIN_ROOT=1 \
  -e LEIN_REPL_PORT=$REPL_PORT \
  -v /$HOME/.m2:/home/$USER/.m2 \
  -v /$HOME/.lein:/home/$USER/.lein \
  -v $(pwd)/:/tvm \
  --net=host -w /tvm \
  $IMG $@

