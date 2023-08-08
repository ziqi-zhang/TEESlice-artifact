import os
import sys
from pdb import set_trace as st
import numpy as np
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
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.linear_shares import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_freivalds_conv import SGXFrevaldsConvBase
from python.layers.quant_relu import SecretEnclaveQuantReLULayer
from python.basic_utils import ExecutionModeOptions

from python.logger_utils import Logger
from python.quantize_net import NetQ
from python.test_linear_shares import argparser_distributed, marshal_process, load_cifar10, seed_torch
from python.timer_utils import NamedTimerInstance, VerboseLevel, NamedTimer
from python.torch_utils import compare_expected_actual

device_cuda = torch.device("cuda:0")
torch.set_printoptions(precision=10)
def compare_layer_member(layer: SecretLinearLayerBase, layer_name: str,
                         extract_func , member_name: str, save_path=None) -> None:
    print(member_name)
    layer.make_sure_cpu_is_latest(member_name)
    compare_expected_actual(extract_func(layer_name), layer.get_cpu(member_name), get_relative=True, verbose=True)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Directory ", save_path, " Created ")
        else:
            print("Directory ", save_path, " already exists")

        torch.save(extract_func(layer_name), os.path.join(save_path, member_name + "_expected"))
        torch.save(layer.get_cpu(member_name), os.path.join(save_path, member_name + "_actual"))


def compare_layer(layer: SecretLinearLayerBase, layer_name: str, save_path=None) -> None:
    print("comparing with layer in expected NN :", layer_name)
    compare_name_function = [("input", get_layer_input), ("output", get_layer_output),
                             ("DerOutput", get_layer_output_grad), ]
    if layer_name != "conv1":
        compare_name_function.append(("DerInput", get_layer_input_grad))
    for member_name, extract_func in compare_name_function:
        compare_layer_member(layer, layer_name, extract_func, member_name, save_path=save_path)

def compare_weight_layer(layer: SecretLinearLayerBase, layer_name: str, save_path=None) -> None:
    compare_layer(layer, layer_name, save_path)
    compare_name_function = [("weight", get_layer_weight), ("DerWeight", get_layer_weight_grad) ]
    for member_name, extract_func in compare_name_function:
        compare_layer_member(layer, layer_name, extract_func, member_name, save_path=save_path)

import sys
import pdb
from pdb import set_trace as st

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
        from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def test_quant_relu(
    batch_size, img_hw, input_c, output_c,
    set_values_to_one=False,
    sid=0
):
    print("="*20, "TestQuantRelu", "="*20)
    print(
        f"batch {batch_size}, img_hw {img_hw}, input_c {input_c}, output_c {output_c}, "
    )
    
    # batch_size = 128
    # input_c = 3
    # output_c = 64
    # img_hw = 224
    # kernel, padding, stride = 7, 3, 2

    # batch_size = 128
    # input_c = 512
    # output_c = 512
    # img_hw = 7
    # kernel, padding, stride = 3, 1, 1

    x_shape = [batch_size, input_c, img_hw, img_hw]

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape, ExecutionModeOptions.GPU)
    input = torch.rand(x_shape) - 0.3
    num_elem = torch.numel(input)
    print("Input: ")
    print(input)
    print("\n\n")
    if set_values_to_one:
        input.zero_()
        input += 0.5
    # print("input: ", input)

    test_layer = SecretEnclaveQuantReLULayer(
        sid, f"TestQuantRelu", ExecutionModeOptions.Enclave
    )
    # test_layer = SecretReLULayer(
    #     sid, f"TestQuantRelu", ExecutionModeOptions.Enclave
    # )
    

    output_layer = SecretOutputLayer(sid, "OutputLayer", ExecutionModeOptions.Enclave, inference=True)
    layers = [input_layer, test_layer, output_layer]
    
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)
    
    input_layer.StoreInEnclave = False

    plain_module = nn.ReLU()
    # num_elem = torch.numel(plain_module.weight.data)
    # plain_module.weight.data = torch.arange(num_elem).reshape(plain_module.weight.shape).float() 
    # print("Pytorch weight \n", plain_module.weight.data)

    
    # print("Bias: ", plain_module.bias.data[:10])
    plain_output = plain_module(input)

    input_layer.set_input(input)
    secret_nn.forward()
    secret_nn.plain_forward()
    secret_output = output_layer.get_cpu("input")

    test_layer.show_plain_error_forward()
    

    output = plain_module(input)

    GlobalTensor.destroy()



if __name__ == "__main__":
    # sys.stdout = Logger()

    seed_torch(123)
    # test_linear()

    test_quant_relu(
        batch_size=1, img_hw=4, input_c=4, output_c=4,
    )


    # test_conv(
    #     batch_size=64, img_hw=32, input_c=256, output_c=512,
    #     kernel=1, padding=0, stride=2
    # )

    # test_conv(
    #     batch_size=128, img_hw=56, input_c=64, output_c=64,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=56, input_c=64, output_c=64,
    #     kernel=3, padding=1, stride=2
    # )

    # test_conv(
    #     batch_size=128, img_hw=28, input_c=64, output_c=128,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=28, input_c=128, output_c=128,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=28, input_c=64, output_c=128,
    #     kernel=3, padding=1, stride=2
    # )

    # test_conv(
    #     batch_size=128, img_hw=14, input_c=128, output_c=256,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=14, input_c=256, output_c=256,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=14, input_c=128, output_c=256,
    #     kernel=3, padding=1, stride=2
    # )

    # test_conv(
    #     batch_size=128, img_hw=7, input_c=256, output_c=512,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=7, input_c=512, output_c=512,
    #     kernel=3, padding=1, stride=1
    # )
    # test_conv(
    #     batch_size=128, img_hw=7, input_c=256, output_c=512,
    #     kernel=3, padding=1, stride=2
    # )

    # test_conv(
    #     batch_size=128, img_hw=56, input_c=64, output_c=64,
    #     kernel=1, padding=0, stride=1
    # )
    # test_conv(
    #     batch_size=1, img_hw=56, input_c=64, output_c=256,
    #     kernel=1, padding=0, stride=1
    # )

    # test_conv()
