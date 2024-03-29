import os
import sys
from pdb import set_trace as st

import torch
from torch import optim, nn
import torch.distributed as dist

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

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)

class SecretBasicBlock():
    expansion = 1

    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, downsample_layers=None, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBasicBlock, self).__init__()
        self.conv1 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv1",
            n_output_channel=planes, filter_hw=3, stride=stride, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn1", EnclaveMode=EnclaveMode,
        )
        self.relu1 = SecretReLULayer(sid, f"{name_prefix}.relu1", EnclaveMode=EnclaveMode)

        self.conv2 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv2",
            n_output_channel=planes, filter_hw=3, stride=1, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn2", EnclaveMode=EnclaveMode, 
            manually_register_next=True, link_next=False,
        )
        self.relu2 = SecretReLULayer(sid, f"{name_prefix}.relu2", EnclaveMode=EnclaveMode)

        self.add = SecretAddLayer(
            sid=sid, LayerName=f"{name_prefix}.add", EnclaveMode=EnclaveMode,
            manually_register_prev=True
        )
        self.add.register_prev_layer(self.bn2)
        self.bn2.register_next_layer(self.add)

        layers = [
            self.conv1, self.bn1, self.relu1, self.conv2, self.bn2,
        ]
        self.downsample_layers = downsample_layers
        if downsample_layers is not None:
            layers += self.downsample_layers
            self.add.register_prev_layer(self.downsample_layers[-1])
            self.downsample_layers[-1].register_next_layer(self.add)
        layers.append(self.add)
        layers.append(self.relu2)
        self.layers = layers

        # layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2]
        # self.layers = layers

    def __str__(self):
        info = f"SecretBasicBlock\n"
        info += f"\t{self.conv1.LayerName}: {self.conv1}\n"
        info += f"\t{self.bn1.LayerName}: {self.bn1}\n"
        info += f"\t{self.relu1.LayerName}: {self.relu1}"
        info += f"\t{self.conv2.LayerName}: {self.conv2}\n"
        info += f"\t{self.bn2.LayerName}: {self.bn2}\n"
        if self.downsample_layers is not None:
            if len(self.downsample_layers) == 1:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}\n"
            elif len(self.downsample_layers) == 2:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}"
                info += f"\t{self.downsample_layers[1].LayerName}: {self.downsample_layers[1]}\n"
        info += f"\t{self.add.LayerName}: {self.add}\n"
        info += f"\t{self.relu2.LayerName}: {self.relu2}"
        return info
    
    def __repr__(self):
        return self.__str__()

class SecretBottleneck():
    expansion = 4

    def __init__(self, inplanes, planes, sid, name_prefix, stride=1, downsample_layers=None, EnclaveMode=ExecutionModeOptions.Enclave):
        super(SecretBottleneck, self).__init__()
        self.stride = stride

        self.conv1 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv1",
            n_output_channel=planes, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn1 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn1", EnclaveMode=EnclaveMode,
        )
        self.relu1 = SecretReLULayer(sid, f"{name_prefix}.relu1", EnclaveMode=EnclaveMode)

        self.conv2 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv2",
            n_output_channel=planes, filter_hw=3, stride=stride, padding=1, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn2 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn2", EnclaveMode=EnclaveMode,
        )
        self.relu2 = SecretReLULayer(sid, f"{name_prefix}.relu2", EnclaveMode=EnclaveMode)

        self.conv3 = SGXConvBase(
            sid=sid, LayerName=f"{name_prefix}.conv3",
            n_output_channel=planes * self.expansion, filter_hw=1, stride=1, padding=0, 
            EnclaveMode=EnclaveMode, bias=False
        )
        self.bn3 = SecretBatchNorm2dLayer(
            sid=sid, LayerName=f"{name_prefix}.bn3", EnclaveMode=EnclaveMode,
            manually_register_next=True, link_next=False,
        )
        self.relu3 = SecretReLULayer(sid, f"{name_prefix}.relu3", EnclaveMode=EnclaveMode)

        self.add = SecretAddLayer(
            sid=sid, LayerName=f"{name_prefix}.add", EnclaveMode=EnclaveMode,
            manually_register_prev=True
        )
        self.add.register_prev_layer(self.bn3)
        self.bn3.register_next_layer(self.add)

        layers = [
            self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3
        ]
        self.downsample_layers = downsample_layers
        if downsample_layers is not None:
            layers += self.downsample_layers
            self.add.register_prev_layer(self.downsample_layers[-1])
            self.downsample_layers[-1].register_next_layer(self.add)
        layers.append(self.add)
        layers.append(self.relu3)

        # layers = [
        #     self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3
        # ]
        self.layers = layers
        

    #     self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    #     self.bn1 = nn.BatchNorm2d(planes)
    #     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
    #                            padding=1, bias=False)
    #     self.bn2 = nn.BatchNorm2d(planes)
    #     self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
    #     self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    #     self.relu = nn.ReLU(inplace=True)
    #     self.downsample = downsample
    #     self.stride = stride

    # def forward(self, x):
    #     residual = x

    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)

    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = self.relu(out)

    #     out = self.conv3(out)
    #     out = self.bn3(out)

    #     if self.downsample is not None:
    #         residual = self.downsample(x)

    #     out += residual
    #     out = self.relu(out)

    #     return out

    def __str__(self):
        info = f"SecretBottleneck\n"
        info += f"\t{self.conv1.LayerName}: {self.conv1}\n"
        info += f"\t{self.bn1.LayerName}: {self.bn1}\n"
        info += f"\t{self.relu1.LayerName}: {self.relu1}"
        info += f"\t{self.conv2.LayerName}: {self.conv2}\n"
        info += f"\t{self.bn2.LayerName}: {self.bn2}\n"
        info += f"\t{self.relu2.LayerName}: {self.relu2}"
        info += f"\t{self.conv3.LayerName}: {self.conv3}\n"
        info += f"\t{self.bn3.LayerName}: {self.bn3}\n"
        if self.downsample_layers is not None:
            if len(self.downsample_layers) == 1:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}\n"
            elif len(self.downsample_layers) == 2:
                info += f"\t{self.downsample_layers[0].LayerName}: {self.downsample_layers[0]}"
                info += f"\t{self.downsample_layers[1].LayerName}: {self.downsample_layers[1]}\n"
        info += f"\t{self.add.LayerName}: {self.add}\n"
        info += f"\t{self.relu3.LayerName}: {self.relu3}"
        return info
    
    def __repr__(self):
        return self.__str__()

class SecretNNResNet(nn.Module):
    # def __init__(self, block, layers, num_classes=1000, batch_size=8, sid=0):
    #     self.inplanes = 64
    #     self.batch_size = batch_size
    #     self.img_hw = 8
    #     self.sid = sid
    #     super(SecretNNResNet, self).__init__()

    #     self.input_shape = [batch_size, 3, self.img_hw, self.img_hw]
    #     self.input_layer = SecretInputLayer(sid, "InputLayer", self.input_shape)
    #     self.input_layer.StoreInEnclave = False
        
    #     self.conv1 = SGXConvBase(
    #         sid, f"conv1",
    #         self.inplanes, filter_hw=7, stride=2, padding=3, is_enclave_mode=True, 
    #         bias=False
    #     )
        
    #     self.bn1 = SecretBatchNorm2dLayer(sid, f"bn1",is_enclave_mode=True)
    #     self.relu1 = SecretReLULayer(sid, f"relu1",is_enclave_mode=False)
    #     self.maxpool = SecretMaxpool2dLayer(sid, f"maxpool1", filter_hw=7, stride=2, padding=3,is_enclave_mode=False)
    #     self.sgx_layers = [
    #         # self.input_layer, self.conv1, self.bn1, self.relu1, self.maxpool
    #         self.input_layer, self.conv1, self.bn1
    #     ]
    #     self.output_layer = SecretOutputLayer(sid, "OutputLayer", inference=True)
    #     self.sgx_layers.append(self.output_layer)

    def __init__(self, block, layers, num_classes=1000, batch_size=64, sid=0, EnclaveMode=ExecutionModeOptions.Enclave):
        self.inplanes = 64
        self.batch_size = batch_size
        self.img_hw = 224
        self.sid = sid
        super(SecretNNResNet, self).__init__()

        self.input_shape = [batch_size, 3, self.img_hw, self.img_hw]
        self.input_layer = SecretInputLayer(sid, "InputLayer", self.input_shape, ExecutionModeOptions.CPU)
        self.input_layer.StoreInEnclave = False
        
        header_enclave_mode = ExecutionModeOptions.CPU if EnclaveMode is ExecutionModeOptions.Enclave else EnclaveMode
        self.conv1 = SGXConvBase(
            sid, f"conv1",header_enclave_mode, 
            self.inplanes, filter_hw=7, stride=2, padding=3, 
            bias=False
        )
        
        self.bn1 = SecretBatchNorm2dLayer(sid, f"bn1",EnclaveMode=header_enclave_mode)
        self.relu1 = SecretReLULayer(sid, f"relu1",header_enclave_mode)
        self.maxpool = SecretMaxpool2dLayer(sid, f"maxpool1", filter_hw=3, stride=2, padding=1,EnclaveMode=header_enclave_mode)
        self.sgx_layers = [
            self.input_layer, self.conv1, self.bn1, self.relu1, self.maxpool
            # self.input_layer, self.conv1, self.bn1, self.relu1
            # self.input_layer, self.conv1, self.bn1
        ]
        # self.output_layer = SecretOutputLayer(sid, "OutputLayer", inference=True)
        # self.sgx_layers.append(self.output_layer)

        self.layer1_blocks, self.layer1 = self._make_layer(
            block, 64, layers[0], sid=sid, prev_layer=self.maxpool, name_prefix="Layer1", EnclaveMode=EnclaveMode)
        self.sgx_layers += self.layer1

        self.layer2_blocks, self.layer2 = self._make_layer(
            block, 128, layers[1], sid=sid, prev_layer=self.layer1[-1], name_prefix="Layer2", stride=2, EnclaveMode=EnclaveMode)
        self.sgx_layers += self.layer2

        self.layer3_blocks, self.layer3 = self._make_layer(
            block, 256, layers[2], sid=sid, prev_layer=self.layer2[-1], name_prefix="Layer3", stride=2, EnclaveMode=EnclaveMode)
        self.sgx_layers += self.layer3

        self.layer4_blocks, self.layer4 = self._make_layer(
            block, 512, layers[3], sid=sid, prev_layer=self.layer3[-1], name_prefix="Layer4", stride=2, EnclaveMode=EnclaveMode)
        self.sgx_layers += self.layer4

        self.avgpool = SecretAvgpool2dLayer(sid, f"avgpool", filter_hw=7, stride=1, padding=0,EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.avgpool)
        self.flatten = SecretFlattenLayer(sid, f"flatten", EnclaveMode=ExecutionModeOptions.CPU)
        self.sgx_layers.append(self.flatten)
        
        self.fc = SGXLinearBase(sid, f"fc", EnclaveMode, batch_size, num_classes, 512 * block.expansion)
        self.sgx_layers.append(self.fc)

        self.output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.CPU, inference=True)
        self.sgx_layers.append(self.output_layer)


        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # # self.avgpool = nn.AvgPool2d(4, stride=1)
        # # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, blocks, prev_layer, sid, name_prefix, 
        stride=1, dropout=False, EnclaveMode=ExecutionModeOptions.Enclave):
        downsample_layer = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample_conv = SGXConvBase(
                sid=self.sid, LayerName=f"{name_prefix}.0.downsample.conv",
                n_output_channel=planes * block.expansion, filter_hw=1, stride=stride, padding=0, 
                EnclaveMode=EnclaveMode, bias=False, manually_register_prev=True
            )
            downsample_conv.register_prev_layer(prev_layer)
            downsample_bn = SecretBatchNorm2dLayer(
                sid=self.sid, LayerName=f"{name_prefix}.0.downsample.bn", 
                EnclaveMode=EnclaveMode, manually_register_next=True, link_next=False,
            )
            downsample_layers = [downsample_conv, downsample_bn]
        else:
            downsample_conv = SecretIdentityLayer(
                sid=self.sid, LayerName=f"{name_prefix}.identity", EnclaveMode=EnclaveMode,
                manually_register_prev=True, manually_register_next=True, link_next=False,
            )
            downsample_conv.register_prev_layer(prev_layer)
            downsample_layers = [downsample_conv]

        layers, layer_blocks = [], []
        start_block = block(
            self.inplanes, planes, sid=sid, name_prefix=f"{name_prefix}.0", 
            stride=stride, downsample_layers=downsample_layers, EnclaveMode=EnclaveMode)
        layer_blocks.append(start_block)
        layers += start_block.layers
        self.inplanes = planes * block.expansion

        prev_layer = start_block.layers[-1]
        for i in range(1, blocks):
            identity = SecretIdentityLayer(
                sid=self.sid, LayerName=f"{name_prefix}.{i}.identity", EnclaveMode=EnclaveMode,
                manually_register_prev=True, manually_register_next=True, link_next=False,
            )
            identity.register_prev_layer(prev_layer)
            identity_layers = [identity]

            middle_block = block(
                self.inplanes, planes, sid=sid, name_prefix=f"{name_prefix}.{i}", 
                stride=1, downsample_layers=identity_layers, EnclaveMode=EnclaveMode
            )
            layer_blocks.append(middle_block)
            layers += middle_block.layers
            prev_layer = middle_block.layers[-1]

        return layer_blocks, layers

    def inject_params(self, pretrained):
        self.conv1.inject_params(pretrained.conv1)
        self.bn1.inject_params(pretrained.bn1)

        def inject_one_block(block, pretrained_block):
            if isinstance(block, SecretBasicBlock):
                block.conv1.inject_params(pretrained_block.conv1)
                block.bn1.inject_params(pretrained_block.bn1)
                block.conv2.inject_params(pretrained_block.conv2)
                block.bn2.inject_params(pretrained_block.bn2)
                if len(block.downsample_layers) > 1:
                    block.downsample_layers[0].inject_params(pretrained_block.downsample[0])
                    block.downsample_layers[1].inject_params(pretrained_block.downsample[1])
            if isinstance(block, SecretBottleneck):
                block.conv1.inject_params(pretrained_block.conv1)
                block.bn1.inject_params(pretrained_block.bn1)
                block.conv2.inject_params(pretrained_block.conv2)
                block.bn2.inject_params(pretrained_block.bn2)
                block.conv3.inject_params(pretrained_block.conv3)
                block.bn3.inject_params(pretrained_block.bn3)
                if len(block.downsample_layers) > 1:
                    block.downsample_layers[0].inject_params(pretrained_block.downsample[0])
                    block.downsample_layers[1].inject_params(pretrained_block.downsample[1])
        
        for i in range(len(self.layer1_blocks)):
            inject_one_block(self.layer1_blocks[i], pretrained.layer1[i])

        for i in range(len(self.layer2_blocks)):
            inject_one_block(self.layer2_blocks[i], pretrained.layer2[i])

        for i in range(len(self.layer3_blocks)):
            inject_one_block(self.layer3_blocks[i], pretrained.layer3[i])

        for i in range(len(self.layer4_blocks)):
            inject_one_block(self.layer4_blocks[i], pretrained.layer4[i])

        self.fc.inject_params(pretrained.fc)




def secret_resnet18(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNResNet(SecretBasicBlock, [2, 2, 2, 2], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def secret_resnet34(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNResNet(SecretBasicBlock, [3, 4, 6, 3], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def secret_resnet50(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # resnet_constructor = SecretNNResNet(SecretBottleneck, [2, 2, 2, 2], **kwargs)
    resnet_constructor = SecretNNResNet(SecretBottleneck, [3, 4, 6, 3], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

def secret_resnet101(pretrained=False, EnclaveMode=ExecutionModeOptions.Enclave, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_constructor = SecretNNResNet(SecretBottleneck, [3, 4, 23, 3], EnclaveMode=EnclaveMode, **kwargs)
    return resnet_constructor

if __name__=="__main__":
    from .resnet import *
    

    # new_conv = nn.Conv2d(3, 1, kernel_size=7, stride=2, padding=3, bias=False)
    # new_conv.weight.data.copy_(pretrained.conv1.weight.data[:1])
    # pretrained.conv1 = new_conv

    
    GlobalTensor.init()
    resnet_constructor = secret_resnet18(EnclaveMode=ExecutionModeOptions.GPU)
    # resnet_constructor = secret_resnet34(EnclaveMode=ExecutionModeOptions.Enclave)
    # resnet_constructor = secret_resnet50(EnclaveMode=ExecutionModeOptions.Enclave)
    # resnet_constructor = secret_resnet101(EnclaveMode=ExecutionModeOptions.Enclave)
    layers = resnet_constructor.sgx_layers

    pretrained = resnet18(pretrained=True)
    # pretrained = resnet34(pretrained=True)
    # pretrained = resnet50(pretrained=True)
    # pretrained = resnet101(pretrained=True)
    pretrained.eval()

    secret_nn = SecretNeuralNetwork(0, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    

    for layer in layers:
        layer.print_connection_info()
    layers[0].print_tensor_link_relation()

    resnet_constructor.inject_params(pretrained)


    input_shape = resnet_constructor.input_shape
    input = torch.rand(input_shape) 

    layers[0].set_input(input)

    secret_nn.forward()
    secret_output = resnet_constructor.output_layer.get_prediction()
    # layers[2].plain_forward()
    # secret_output = layers[2].PlainForwardResult
    # st()
    # secret_output = layers[2].get_cpu("input")


    plain_output = pretrained(input)
    # plain_output = plain_module(input)

    # secret_middle = resnet_constructor.layer2_blocks[0].conv1.get_cpu("input")
    # err = compare_expected_actual(
    #     expected=middle, actual=secret_middle
    # )
    # print(f"Forward Error: {err}")
    # st()

    err = compare_expected_actual(
        expected=plain_output, actual=secret_output
    )
    print(f"Forward Error: {err}")

