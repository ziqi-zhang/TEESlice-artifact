# Goten
**GPU-Outsourcing Trusted Execution of Neural Network Training**

A framework leveraging GPU and Intel SGX to protect privacy of training data, model, and queries while achieving high-performance training and prediction

The details of this project are presented in the following paper:

[*Goten: GPU-Outsourcing Trusted Execution of Neural Network Training*](https://lucieno.github.io/files/goten.pdf) </br>
**Lucien K. L. Ng, Sherman S. M. Chow, Anna P. Y. Woo, Donald P. H. Wong, Yongjun Zhao** </br>
To appear in AAAI-21

## How to Install

1. Install Intel's [linux-sgx](https://github.com/intel/linux-sgx) and [linux-sgx-driver](https://github.com/intel/linux-sgx-driver).
    We tested our code on SGX SDK 2.6
    
1. Install PyTorch with Python3. You may install it using [Anaconda](https://www.anaconda.com/)

        conda create -n goten python=3.6
        conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch
        
1. Make the C++ part of this repo

        make -j4

## How to run
1. Source your Intel SGK SDK environment. For example

        source /opt/intel/sgxsdk/environment

1. Run VGG11

    If you want to run all 3 non-colluding servers in a local machine, run the following command

        python -m python.vgg
        
    If the servers are distributed on different machines, please mark them as S0, S1, and S2, then
    
    - S0 run the following command, where IPS0 is the IP address of S0
    
            python -m python.vgg --ip="$IPS0" -s 0
    
    - For S1
    
            python -m python.vgg --ip="$IPS0" -s 1
    
    - For S2
    
            python -m python.vgg --ip="$IPS0" -s 2
            
## CaffeSCONE

The guide of building and running CaffeSCONE is store in the folder named [CaffeSCONE](CaffeSCONE).

## Disclaimer
DO NOT USE THIS SOFTWARE TO SECURE ANY REAL-WORLD DATA OR COMPUTATION!

This software is a proof-of-concept meant for performance testing of the Goten framework ONLY. It is full of security vulnerabilities that facilitate testing, debugging and performance measurements. In any real-world deployment, these vulnerabilities can be easily exploited to leak all user inputs.

Some parts that have a negligble impact on performance but that are required for a real-world deployment are not currently implemented (e.g., setting up a secure communication channel with a remote client and producing verifiable attestations).

## Acknowledgement
we reuse some code of Slalom (Tramer & Boneh, 2019), 
including their code of crypgtographicially-secure random number generation and encryption/decryption, 
and their OS-call-free version of Eigen, 
a linear algebra library.
