# CaffeSCONE

Caffe is one of the state-of-the-art DNN libraries written in C++.
We deploy Caffe into a secure linux container framework -- SCONE, 
which allows DNNs run under the protection of intel SGX. 

As SGX and SCONE introduce performance overhead,
we want to measure the performance of Caffe running inside SCONE compared to it running in plaintext CPU mode.

## Building the Framework

CaffeSCONE is exported to a docker image. 
To build the framework, you can use a linux machine and install docker. 
We build the framework and ran our experiments on a machine running Ubuntu 16.04 LTS.

1. Install Docker:

        sudo apt-get update
        sudo apt-get install docker.io

1. Pull the Docker image and run it:

        sudo docker pull donald1119/caffescone:latest
        sudo docker run -it donald1119/caffescone:latest

## Exemplary Experiments

1. Go to the root directory of Caffe:

        cd /caffe

1. Set the number of threads using to run OpenBLAS(e.g., 1):

        export OPENBLAS_NUM_THREADS=1

1. Benchmarking the time spending on each layer:

    To run the experiment using SCONE cross-compiler, 
    you need to set the following variables at the beginning of the command we are going to run:

    - SCONE_VERSION: set to 1 to print out the log message of SCONE cross compiler

    - SCONE_HEAP: the amount the heap memory to allocated to run the task

    - SCONE_STACK: the amount the stack memory to allocated to run the task

    - SCONE_MODEHW: for hardware mode; SIM for simulation mode

    The following experiments are running the Cifar10 dataset to perform DNN task on VGG9, 
    the .prototxt and dataset are located in /caffe/examples/cifar10. 
    You can modify the batch size in cifar10_vgg9_train_test.prototxt

        SCONE_VERSION=1 SCONE_HEAP=4000M SCONE_STACK=256M SCONE_MODE=SIM ./build/tools/caffe time -model examples cifar10/cifar10_vgg9_train_test.prototxt -iterations 5

1. To record the time spent on the solver in each iteration during the training:

        SCONE_VERSION=1 SCONE_HEAP=4000M SCONE_STACK=256M SCONE_MODE=SIM ./build/tools/caffe train --solver=examples/cifar10/cifar10_vgg9_solver.prototxt
