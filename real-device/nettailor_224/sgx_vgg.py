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

class SecretNNVGG16BN(nn.Module):

    def __init__(self, num_classes=1000, batch_size=1, sid=0, EnclaveMode=ExecutionModeOptions.Enclave):
        self.inplanes = 64
        self.batch_size = batch_size
        self.img_hw = 224
        self.sid = sid
        super(SecretNNVGG16BN, self).__init__()

        self.input_shape = [batch_size, 3, self.img_hw, self.img_hw]
        self.input_layer = SecretInputLayer(sid, "InputLayer", self.input_shape, EnclaveMode)
        self.input_layer.StoreInEnclave = False
        
        self.conv1 = SGXConvBase(
            sid, f"conv1",EnclaveMode, 
            64, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(sid, f"bn1",EnclaveMode)
        self.relu1 = SecretReLULayer(sid, f"relu1", EnclaveMode)

        self.conv2 = SGXConvBase(
            sid, f"conv2",EnclaveMode, 
            64, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(sid, f"bn2",EnclaveMode)
        self.relu2 = SecretReLULayer(sid, f"relu2", EnclaveMode)
        self.maxpool2 = SecretMaxpool2dLayer(sid, f"maxpool2", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.conv3 = SGXConvBase(
            sid, f"conv3",EnclaveMode, 
            128, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn3 = SecretBatchNorm2dLayer(sid, f"bn3",EnclaveMode)
        self.relu3 = SecretReLULayer(sid, f"relu3", EnclaveMode)

        self.conv4 = SGXConvBase(
            sid, f"conv4",EnclaveMode, 
            128, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn4 = SecretBatchNorm2dLayer(sid, f"bn4",EnclaveMode)
        self.relu4 = SecretReLULayer(sid, f"relu4", EnclaveMode)
        self.maxpool4 = SecretMaxpool2dLayer(sid, f"maxpool4", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.conv5 = SGXConvBase(
            sid, f"conv5",EnclaveMode, 
            256, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn5 = SecretBatchNorm2dLayer(sid, f"bn5",EnclaveMode)
        self.relu5 = SecretReLULayer(sid, f"relu5", EnclaveMode)

        self.conv6 = SGXConvBase(
            sid, f"conv6",EnclaveMode, 
            256, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn6 = SecretBatchNorm2dLayer(sid, f"bn6",EnclaveMode)
        self.relu6 = SecretReLULayer(sid, f"relu6", EnclaveMode)

        self.conv7 = SGXConvBase(
            sid, f"conv7",EnclaveMode, 
            256, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn7 = SecretBatchNorm2dLayer(sid, f"bn7",EnclaveMode)
        self.relu7 = SecretReLULayer(sid, f"relu7", EnclaveMode)
        self.maxpool7 = SecretMaxpool2dLayer(sid, f"maxpool7", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.conv8 = SGXConvBase(
            sid, f"conv8",EnclaveMode, 
            512, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn8 = SecretBatchNorm2dLayer(sid, f"bn8",EnclaveMode)
        self.relu8 = SecretReLULayer(sid, f"relu8", EnclaveMode)

        self.conv9 = SGXConvBase(
            sid, f"conv9",EnclaveMode, 
            512, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn9 = SecretBatchNorm2dLayer(sid, f"bn9",EnclaveMode)
        self.relu9 = SecretReLULayer(sid, f"relu9", EnclaveMode)

        self.conv10 = SGXConvBase(
            sid, f"conv10",EnclaveMode, 
            512, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn10 = SecretBatchNorm2dLayer(sid, f"bn10",EnclaveMode)
        self.relu10 = SecretReLULayer(sid, f"relu10", EnclaveMode)
        self.maxpool10 = SecretMaxpool2dLayer(sid, f"maxpool10", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.conv11 = SGXConvBase(
            sid, f"conv11",EnclaveMode, 
            512, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn11 = SecretBatchNorm2dLayer(sid, f"bn11",EnclaveMode)
        self.relu11 = SecretReLULayer(sid, f"relu11", EnclaveMode)

        self.conv12 = SGXConvBase(
            sid, f"conv12",EnclaveMode, 
            512, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn12 = SecretBatchNorm2dLayer(sid, f"bn12",EnclaveMode)
        self.relu12 = SecretReLULayer(sid, f"relu12", EnclaveMode)

        self.conv13 = SGXConvBase(
            sid, f"conv13",EnclaveMode, 
            512, filter_hw=3, stride=1, padding=1, 
            bias=False
        )
        self.bn13 = SecretBatchNorm2dLayer(sid, f"bn13",EnclaveMode)
        self.relu13 = SecretReLULayer(sid, f"relu13", EnclaveMode)
        self.maxpool13 = SecretMaxpool2dLayer(sid, f"maxpool13", filter_hw=2, stride=2, padding=0,EnclaveMode=EnclaveMode)

        self.sgx_layers = [
            self.input_layer, 
            self.conv1, self.bn1, self.relu1, 
            self.conv2, self.bn2, self.relu2, self.maxpool2,
            self.conv3, self.bn3, self.relu3,
            self.conv4, self.bn4, self.relu4, self.maxpool4,
            self.conv5, self.bn5, self.relu5,
            self.conv6, self.bn6, self.relu6, 
            self.conv7, self.bn7, self.relu7, self.maxpool7,
            self.conv8, self.bn8, self.relu8,
            self.conv9, self.bn9, self.relu9, 
            self.conv10, self.bn10, self.relu10, self.maxpool10,
            self.conv11, self.bn11, self.relu11,
            self.conv12, self.bn12, self.relu12, 
            self.conv13, self.bn13, self.relu13, self.maxpool13,
        ]
        
        self.avgpool = SecretAvgpool2dLayer(sid, f"avgpool", filter_hw=7, stride=7, padding=0,EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.avgpool)
        self.flatten = SecretFlattenLayer(sid, f"flatten", EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.flatten)
        
        self.fc = SGXLinearBase(sid, f"fc", EnclaveMode, batch_size, num_classes, 512)
        self.sgx_layers.append(self.fc)

        self.output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.CPU, inference=True)
        self.sgx_layers.append(self.output_layer)




def secret_vgg16_bn(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNVGG16BN( EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor


if __name__=="__main__":
    from .pytorch_resnet import *
    
    batch_size = 16
    iteration = 5
    torch.set_num_threads(1)

    
    GlobalTensor.init()
    resnet_constructor = secret_vgg16_bn(
        EnclaveMode=ExecutionModeOptions.Enclave, 
        batch_size=batch_size,
        
    )
    layers = resnet_constructor.sgx_layers

    pretrained = resnet18(pretrained=True)
    # pretrained = resnet34(pretrained=True)
    pretrained.eval()

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
