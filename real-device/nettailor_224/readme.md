# 概述
测各个模型在真实SGX场景下的速度，输入图片是224x224

# 计算nettailor的时间和吞吐率结果
- 训练好的模型保存在nettailor_224/models/nettailor中
- sgx_student_resnet_nettailor_freivalds.py 加载 nettailor的checkpoint并且inference
- eval_sgx_nettailor_freivalds.py 和 eval_sgx_nettailor_freivalds.sh 对所有模型和数据集进行遍历，依次计算每个setting的时间和吞吐率
- 结果保存在nettailor_224/models/nettailor

# nettailor的params和flops
- nettailor_resnet_224.py 是输入数据为224x224前提下的的nettailor  student_resnet
- 通过 ``python -m nettailor_224.nettailor_resnet_224`` 可以计算上面保存模型的task的param和flop

# 计算原来模型的params和flops
- alexnet.py, resnet.py和vgg.py在网络的构造函数都会计算params和flops，打印在屏幕上
- flops是设定输入大小为224x224，计算结果基本与 https://mmclassification.readthedocs.io/en/latest/model_zoo.html 一致

# cifar10的结果
- 包括三个模型 cifar10_alexnet.py, cifar10_resnet.py和cifar10_vgg.py ，但是后来不怎么用了

# 假设ReLU在SGX里面，linear层在GPU上的baseline
- 包括三个文件  sgx_alexnet_relu_in_sgx.py, sgx_resnet_relu_in_sgx.py, sgx_vgg_relu_in_sgx.py，直接``python -m ``就可以单次运行并及时

# Goten/knockoff/nettailor 文件夹是加载模型的时候用的，貌似pytorch会读取原来的文件路径，所以这里放fake路径

# alexnet, vgg 在 goten 代码框架下的实现
- 分别在sgx_alexnet.py, sgx_resnet.py 和 sgx_vgg.py

# pytorch版本下的alexnet, resnet, vgg，可以用在occlum里面
- 分别是pytorch_alexnet.py, pytorch_resnet.py, pytorch_vgg.py
- 测的时候注意关掉多线程  ``torch.set_num_threads(1)``

# 在jetson上跑的代码
- 结果保存在``nettailor_224/models_nettailor/jetson``中
- 几个baseline的代码是``hetero_pytorch_*``
- 在jetson中把nettailor模型跑起来的代码是``hetero_student_all.py``


# 20230223 再次尝试在真实设备上跑结果
- Shield-whole-model的结果，resnet18，``python -m nettailor_224.sgx_resnet``
    - ``BS 64, ITER 10. Time mean 120.9933 ms, std 9.8267 ms. Throughput mean 8.3177, std 0.6500``
    - ``BS 1, ITER 10. Time mean 159.6795 ms, std 3.8069 ms. Throughput mean 6.2661, std 0.1474``
- ShadowNet 的结果，resnet18, ``python -m nettailor_224.sgx_resnet_relu_in_sgx``, ``EnclaveMode=ExecutionModeOptions.GPU``
    - ``BS 64, ITER 5, time 18.1233, throughput 55.1775``
    - ``EnclaveMode=ExecutionModeOptions.CPU``，即CPU的结果``BS 64, ITER 5, time 60.1420, throughput 16.6273``

- Nettailor, ``bash nettailor_224/eval_sgx_nettailor_freivalds.sh``，只跑CIFAR10, ResNet18
    - ``sgx_student_resnet_nettailor_freivalds``中``resnet_constructor``的配置是``EnclaveMode=ExecutionModeOptions.GPU``
        - BS=64: ``Time mean 41.17, std 0.73. Throughputs mean 24.30, std 0.43``
        - BS=32: ``Time mean 36.97, std 0.81. Throughputs mean 27.06, std 0.59``
        - BS=1:  ``Time mean 63.71, std 1.74. Throughputs mean 15.71, std 0.42``
    - 改成CPU的形式 ``EnclaveMode=ExecutionModeOptions.CPU``
        - ``Time mean 72.81, std 0.75. Throughputs mean 13.74, std 0.14``
    - 直接使用每层layer的时间相加，``TimePerSample = 19.7 ms``, ``Throughput = 50``

- Shield-whole-model的结果，alexnet，``python -m nettailor_224.sgx_alexnet``
    - ``BS 64, ITER 10. Time mean 128.8894 ms, std 19.5491 ms. Throughput mean 7.9111, std 1.0065``



# Shield-whole-model baseline 结果
- resnet18
    - ``BS 64, ITER 10. Time mean 86.3121 ms, std 0.8513 ms. Throughput mean 11.5870, std 0.1147``
    - ``BS 32, ITER 10. Time mean 85.7035 ms, std 1.6139 ms. Throughput mean 11.6723, std 0.2189``
    - ``BS 16, ITER 10. Time mean 89.3003 ms, std 1.5039 ms. Throughput mean 11.2013, std 0.1872``
    - ``BS 1, ITER 10. Time mean 130.4687 ms, std 4.3216 ms. Throughput mean 7.6729, std 0.2469``
    - GPU: ``BS 16, ITER 10. Time mean 3.5749 ms, std 0.1924 ms. Throughput mean 280.5550, std 15.3475``

- alexnet
    - ``BS 64, ITER 10. Time mean 117.9307 ms, std 0.7243 ms. Throughput mean 8.4799, std 0.0519``
    - ``BS 16, ITER 10. Time mean 119.4554 ms, std 2.0676 ms. Throughput mean 8.3738, std 0.1432``
    - ``BS 1, ITER 10. Time mean 152.7595 ms, std 6.2464 ms. Throughput mean 6.5568, std 0.2592``
    - GPU: ``BS 16, ITER 10. Time mean 2.0249 ms, std 0.1083 ms. Throughput mean 495.2676, std 26.3699``

- vgg16
    - ``BS 16, ITER 5. Time mean 644.2232 ms, std 2.5935 ms. Throughput mean 1.5523, std 0.0062``
    - ``BS 1, ITER 5. Time mean 746.2860 ms, std 6.4007 ms. Throughput mean 1.3401, std 0.0115``
    - GPU: ``BS 16, ITER 5. Time mean 9.7032 ms, std 0.2039 ms. Throughput mean 103.1035, std 2.1541``

# 记录所有case的terminal output到文件，结果和分析脚本在``nettailor_224/layer_analysis``里面
- ``bash bash nettailor_224/eval_sgx_nettailor_freivalds_layerwise.sh`` 会把``txt``文件保存在``nettailor_224/layer_analysis``中
- ``layer_analysis.py`` 会从``txt``文件中提取数据，保存在``json``文件中