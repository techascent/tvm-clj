FROM ubuntu:18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    make g++ cmake llvm-dev libopenblas-dev \
    ocl-icd-* opencl-headers \
    openjdk-11-jdk-headless wget curl


RUN wget https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein &&\
    chmod a+x lein &&\
    mv lein /usr/bin

ARG USERID
ARG GROUPID
ARG USERNAME

RUN groupadd -g $GROUPID $USERNAME
RUN useradd -u $USERID -g $GROUPID $USERNAME
RUN mkdir /home/$USERNAME && chown $USERNAME:$USERNAME /home/$USERNAME
