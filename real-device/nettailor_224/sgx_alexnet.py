import os
import sys
from pdb import set_trace as st

import torch
from torch import optim, nn
import torch.distributed as dist
import time
import numpy as np

from python.common_net import register_layer, register_weight_layer, get_layer_weight, get_layer_input, \
    get_layer_weight_grad, get_layer_output, get_layer_output_grad, get_layer_input_grad
from python.enclave_interfaces import GlobalTensor
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.conv2d import SecretConv2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.linear_base import SecretLinearLayerBase
from python.layers.matmul import SecretMatmulLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.linear_shares import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.identity import SecretIdentityLayer
from python.layers.add import SecretAddLayer

from python.logger_utils import Logger
from python.quantize_net import NetQ
from python.test_linear_shares import argparser_distributed, marshal_process, load_cifar10, seed_torch
from python.timer_utils import NamedTimerInstance, VerboseLevel, NamedTimer
from python.torch_utils import compare_expected_actual
from python.basic_utils import ExecutionModeOptions

from nettailor_224.pytorch_alexnet import *

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)

class SecretNNAlexNet(nn.Module):

    def __init__(self, num_classes=1000, batch_size=1, sid=0, EnclaveMode=ExecutionModeOptions.Enclave):
        self.inplanes = 64
        self.batch_size = batch_size
        self.img_hw = 224
        self.sid = sid
        super(SecretNNAlexNet, self).__init__()

        self.input_shape = [batch_size, 3, self.img_hw, self.img_hw]
        self.input_layer = SecretInputLayer(sid, "InputLayer", self.input_shape, EnclaveMode)
        self.input_layer.StoreInEnclave = False
        
        self.conv1 = SGXConvBase(
            sid, f"conv1",EnclaveMode, 
            64, filter_hw=11, stride=2, padding=5, 
            bias=False
        )
        self.relu1 = SecretReLULayer(sid, f"relu1", EnclaveMode)
        self.maxpool1 = SecretMaxpool2dLayer(sid, f"maxpool1", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.conv2 = SGXConvBase(
            sid, f"conv2",EnclaveMode, 
            256, filter_hw=5, stride=1, padding=2, 
            bias=False
        )
        self.relu2 = SecretReLULayer(sid, f"relu2", EnclaveMode)
        self.maxpool2 = SecretMaxpool2dLayer(sid, f"maxpool3", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.conv3 = SGXConvBase(
            sid, f"conv3",EnclaveMode, 
            384, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.relu3 = SecretReLULayer(sid, f"relu3", EnclaveMode)

        self.conv4 = SGXConvBase(
            sid, f"conv4",EnclaveMode, 
            256, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.relu4 = SecretReLULayer(sid, f"relu4", EnclaveMode)

        self.conv5 = SGXConvBase(
            sid, f"conv5",EnclaveMode, 
            256, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.relu5 = SecretReLULayer(sid, f"relu5", EnclaveMode)
        self.maxpool5 = SecretMaxpool2dLayer(sid, f"maxpool5", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)
        self.maxpool6 = SecretMaxpool2dLayer(sid, f"maxpool6", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.sgx_layers = [
            self.input_layer, self.conv1, self.relu1, self.maxpool1, 
            self.conv2, self.relu2, self.maxpool2,
            self.conv3, self.relu3,
            self.conv4, self.relu4,
            self.conv5, self.relu5,
            self.maxpool5, self.maxpool6
        ]
        

        self.avgpool = SecretAvgpool2dLayer(sid, f"avgpool", filter_hw=7, stride=7, padding=0,EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.avgpool)
        self.flatten = SecretFlattenLayer(sid, f"flatten", EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.flatten)
        
        self.fc = SGXLinearBase(sid, f"fc", EnclaveMode, batch_size, num_classes, 256)
        self.sgx_layers.append(self.fc)

        self.output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.CPU, inference=True)
        self.sgx_layers.append(self.output_layer)


def secret_alexnet(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNAlexNet( EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor


if __name__=="__main__":
    from .pytorch_resnet import *
    
    batch_size = 1
    iteration = 10
    torch.set_num_threads(1)

    mode = ExecutionModeOptions.Enclave

    
    GlobalTensor.init()
    resnet_constructor = secret_alexnet(EnclaveMode=mode, batch_size=batch_size)
    layers = resnet_constructor.sgx_layers

    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    
    for layer in layers:
        layer.print_connection_info()
    layers[0].print_tensor_link_relation()

    inference_times, throughputs = [], []
    for i in range(iteration):
        input_shape = resnet_constructor.input_shape
        input = torch.rand(input_shape) 

        layers[0].set_input(input)

        start = time.time()
        secret_nn.forward()
        secret_output = resnet_constructor.output_layer.get_prediction()
        end = time.time()
        elapse = (end-start) * 1000 / batch_size
        # if mode == ExecutionModeOptions.GPU:
        #     elapse -= 140 / batch_size
        inference_times.append(elapse)
        throughput = 1000 / elapse
        throughputs.append(throughput)

    inference_times = inference_times[1:]
    throughputs = throughputs[1:]
        
    inf_mean = np.mean(inference_times)
    inf_std = np.std(inference_times)
    th_mean = np.mean(throughputs)
    th_std = np.std(throughputs)
    print(f"BS {batch_size}, ITER {iteration}. Time mean {inf_mean:.4f} ms, std {inf_std:.4f} ms. Throughput mean {th_mean:.4f}, std {th_std:.4f}")

