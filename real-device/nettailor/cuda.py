import torch
import torch.nn as nn

from pdb import set_trace as st
import functools

from python.common_net import register_layer, register_weight_layer, get_layer_weight, get_layer_input, \
    get_layer_weight_grad, get_layer_output, get_layer_output_grad, get_layer_input_grad
from python.enclave_interfaces import GlobalTensor
from python.layers.base import SecretLayerBase
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.conv2d import SecretConv2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.linear_base import SecretLinearLayerBase
from python.layers.matmul import SecretMatmulLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.linear_shares import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.identity import SecretIdentityLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer

from python.logger_utils import Logger
from python.quantize_net import NetQ
from python.test_linear_shares import argparser_distributed, marshal_process, load_cifar10, seed_torch
from python.timer_utils import NamedTimerInstance, VerboseLevel, NamedTimer
from python.torch_utils import compare_expected_actual
from python.basic_utils import ExecutionModeOptions

def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
        # __str__ = cls.__str__
        # __repr__ = cls.__repr__
        def __str__(self):
            return "==========="
        
        def __repr__(self):
            return self.__str__()

    return NewCls

def test_combination(sid=0, ):
    batch_size = 2
    n_img_channel = 16
    img_hw = 56
    in_feature, out_feature = 512, 100

    # x_shape = [batch_size, n_img_channel, img_hw, img_hw]
    x_shape = [batch_size, in_feature]


    secret_partial_avgpool = partialclass(SecretAvgpool2dLayer, filter_hw=7, stride=1, padding=0)
    plain_partial_avgpool = partialclass(nn.AvgPool2d, kernel_size=7, stride=2, padding=0 )
    secret_partial_maxpool = partialclass(SecretMaxpool2dLayer, filter_hw=3, stride=2, padding=1)
    plain_partial_maxpool = partialclass(nn.MaxPool2d, kernel_size=3, stride=2, padding=1)
    plain_partial_bn = partialclass(nn.BatchNorm2d, num_features=n_img_channel)
    secret_partial_conv = partialclass(
        SGXConvBase, n_output_channel=n_img_channel, filter_hw=3, stride=1, padding=1,
        batch_size=batch_size, n_input_channel=n_img_channel, img_hw=img_hw, bias=False)
    plain_partial_conv = partialclass(
        nn.Conv2d, in_channels=n_img_channel, out_channels=n_img_channel, 
        kernel_size=3, stride=1, padding=1, bias=False
    )
    secret_partial_linear = partialclass(
        SGXLinearBase, n_output_features=out_feature, n_input_features=in_feature, batch_size=batch_size
    )
    plain_partial_linear = partialclass(
        nn.Linear, in_features=in_feature, out_features=out_feature
    )

    for layer_pair in [
        # (SecretIdentityLayer, nn.Identity),
        # (SecretReLULayer, nn.ReLU),
        # (secret_partial_avgpool, plain_partial_avgpool)
        # (secret_partial_maxpool, plain_partial_maxpool)
        # (SecretBatchNorm2dLayer, plain_partial_bn)
        # (secret_partial_conv, plain_partial_conv)
        (secret_partial_linear, plain_partial_linear)
    ]:
        for enclave_mode in (
            ExecutionModeOptions.Enclave, 
            ExecutionModeOptions.CPU, 
            ExecutionModeOptions.GPU
        ):
            print("\n\n")
            test_layer_class, plain_layer_class = layer_pair
            test_layer_repr = test_layer_class

                
            print("Executing ", test_layer_repr, enclave_mode)
            
            GlobalTensor.init()

            input_layer = SecretInputLayer(sid, "InputLayer", x_shape, ExecutionModeOptions.CPU)
            input = torch.rand(x_shape) - 0.3
            # input.zero_()
            # input += 1

            output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.CPU, inference=True)
            
            test_layer = test_layer_class(sid=sid, LayerName=f"TestLayer", EnclaveMode=enclave_mode)
            plain_layer = plain_layer_class()
            if isinstance(plain_layer, nn.BatchNorm2d):
                plain_layer.eval()
            


            layers = [input_layer, test_layer, output_layer]
            
            secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            test_layer.inject_params(plain_layer)

            plain_output = plain_layer(input)

            input_layer.set_input(input)
            secret_nn.forward()
            secret_output = output_layer.get_prediction()

            err = compare_expected_actual(plain_output, secret_output)
            print(test_layer_repr, enclave_mode, "Abs error is: ", err)
            GlobalTensor.destroy()

            if err > 1e-4:
                st()



seed_torch(123)
test_combination()