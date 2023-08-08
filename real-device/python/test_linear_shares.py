import os
import sys
import argparse

import numpy as np
import random

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import torchvision
from torchvision.transforms import transforms

from python.layers.batch_norm_1d import SecretBatchNorm1dLayer
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.conv2d import SecretConv2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.matmul import SecretMatmulLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.linear_shares import init_communicate, warming_up_cuda, conv2d_op, SecretNeuralNetwork, SgdOptimizer, \
    conv2d_input_grad_op, conv2d_weight_grad_op
from python.enclave_interfaces import GlobalTensor, SecretEnum
from python.logger_utils import Logger
from python.common_torch import calc_conv2d_output_shape, get_random_uniform, SecretConfig, \
    generate_unquantized_tensor, mod_move_down, modest_magnitude
from python.quantize_net import pre_quantize, post_quantize
from python.torch_utils import compare_expected_actual
from stateless_logger import StatelessLogger


def load_cifar10(batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = batch_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize, ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1,
                                              drop_last=True, worker_init_fn = lambda x: seed_torch(123))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), normalize, ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def marshal_process(sid, master_add, master_port, target_proc, proc_args):
    params = [master_add, master_port] + proc_args
    if sid == -1:
        processes = []
        for proc_rank in range(SecretConfig.worldSize):
            p = Process(target=target_proc, args=[proc_rank] + params)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        proc_rank = sid
        p = Process(target=target_proc, args=[proc_rank] + params)
        p.start()
        p.join()
        p.terminate()




def init_processes(sid, master_addr, master_port):
    # sys.stdout = Logger()
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    layer_name = "SingleLayer"
    batch_size, n_input_channel, n_output_channel, img_hw, filter_hw = 512, 64, 64, 8, 3
    local_shared_conv2d_quantized(sid, layer_name, n_output_channel, filter_hw, batch_size, n_input_channel, img_hw)


def local_shared_conv2d_quantized(sid, master_addr, master_port, layer_name, conv2d_params):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    print(f"batch_size, n_output_channel, n_input_channel, img_hw, filter_hw: "
          f"{batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    padding = 1
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
    y_shape = calc_conv2d_output_shape(x_shape, w_shape, padding)

    GlobalTensor.init()

    layer = SecretConv2dLayer(sid, layer_name, n_output_channel, filter_hw, batch_size, n_input_channel, img_hw)
    layer.set_eid(GlobalTensor.get_eid())
    layer.link_tensors()
    layer.init_shape()
    layer.init(start_enclave=False)

    trainloader, _ = load_cifar10(batch_size)
    AQ = mod_move_down((get_random_uniform(SecretConfig.PrimeLimit, w_shape) - SecretConfig.PrimeLimit/2).type(SecretConfig.dtypeForCpuOp))
    BQ = mod_move_down((get_random_uniform(SecretConfig.PrimeLimit, x_shape) - SecretConfig.PrimeLimit/2).type(SecretConfig.dtypeForCpuOp))
    input_f = BQ
    weight_f = AQ
    der_output_f = torch.from_numpy(np.random.uniform(0, 1, size=y_shape)).type(SecretConfig.dtypeForCpuOp)

    layer_op_name = layer.LayerName + "Forward"
    InputQ, WeightQ = pre_quantize(input_f, weight_f, layer_op_name, 0)

    layer_op_name = layer.LayerName + "BackwardInput"
    OutputGradQ, WeightQ = pre_quantize(der_output_f, weight_f, layer_op_name, 0)

    layer_op_name = layer.LayerName + "BackwardWeight"
    OutputGradQ, InputQ = pre_quantize(der_output_f, input_f, layer_op_name, 0)

    # AQ = mod_move_down(get_random_uniform(SecretConfig.PrimeLimit, w_shape).type(SecretConfig.dtypeForCpuOp))
    # BQ = mod_move_down(get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(SecretConfig.dtypeForCpuOp))
    # der_outputQ = mod_move_down(get_random_uniform(SecretConfig.PrimeLimit, y_shape).type(SecretConfig.dtypeForCpuOp))

    if sid != 2:
        layer.load_tensors(WeightQ, InputQ, OutputGradQ)

    dist.barrier()
    layer.forward(need_quantize=False)
    dist.barrier()
    layer.backward(need_quantize=False)
    if sid != 2:
        layer.plain_forward(quantized_only=True)
        layer.plain_backward(quantized_only=True)
        err, err_grad_input, err_grad_weight = layer.show_plain_error(quantized_only=True)

        # layer.transfer_enclave_to_cpu("input")

    dist.barrier()
    dist.destroy_process_group()


def local_shared_conv2d(sid, master_addr, master_port, layer_name, conv2d_params):
    # sys.stdout = Logger()
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    logger = StatelessLogger(sid)
    logger.info("StartConv2d")
    batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    print(f"batch_size, n_output_channel, n_input_channel, img_hw, filter_hw: "
          f"{batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    logger.info(f"setting: {batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    padding = 1
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
    y_shape = calc_conv2d_output_shape(x_shape, w_shape, padding)

    dirname = "vgg_conv3"

    def randomize_tensor(x):
        rand_size = 128
        x[:, :rand_size] = torch.rand_like(x[:, :rand_size]).type(SecretConfig.dtypeForCpuOp)

    stored_tensor_names = ["DerInput_actual", "DerOutput_actual", "DerWeight_actual", "input_actual", "output_actual",
                           "weight_actual", "DerInput_expected", "DerOutput_expected", "DerWeight_expected",
                           "input_expected", "output_expected", "weight_expected"]

    # stored_tensors = {}
    # for name in stored_tensor_names:
    #     stored_tensors[name] = torch.load(os.path.join(dirname, name)).cpu()

    # trainloader, _ = load_cifar10(batch_size)
    Af = torch.zeros(w_shape)
    torch.nn.init.xavier_normal_(Af, 1)
    # Af = stored_tensors["weight_expected"]
    # Af = torch.ones(w_shape)
    # Bf, _ = next(iter(trainloader))
    Bf = torch.rand(x_shape)
    Bf = torch.ones(x_shape)
    randomize_tensor(Bf)
    # Bf = stored_tensors["input_expected"]
    # Af, Bf = Af.detach(), Bf.detach()
    der_output_f = torch.from_numpy(np.random.uniform(0, 1, size=y_shape)).type(SecretConfig.dtypeForCpuOp)
    der_output_f = torch.ones(y_shape).type(SecretConfig.dtypeForCpuOp)
    randomize_tensor(der_output_f)
    # der_output_f = stored_tensors["DerOutput_expected"]

    GlobalTensor.init()

    input_f = Bf
    weight_f = Af

    layer = SecretConv2dLayer(sid, layer_name, n_output_channel, filter_hw, batch_size, n_input_channel, img_hw)
    layer.init_shape()
    layer.set_eid(GlobalTensor.get_eid())
    layer.link_tensors()
    layer.init(start_enclave=False)
    if sid != 2:
        layer.load_tensors(weight_f, input_f, der_output_f, for_quantized=False)
        # layer.inject_params(q_conv2d)

        print("going to compare Bf")
        layer.ForwardOutput.transfer_enclave_to_cpu("Bf")
        compare_expected_actual(input_f, layer.ForwardOutput.get_cpu("Bf"), get_relative=True, verbose=True)

    dist.barrier()
    layer.forward(need_quantize=True)
    dist.barrier()
    layer.backward(need_quantize=True)
    if sid != 2:
        layer.plain_forward(quantized_only=False)
        layer.plain_backward(quantized_only=False)

        logger.info("EndConv2d")
        # layer.show_plain_error(quantized_only=False)

    # if sid == 1:
    #     layer.ForwardOutput.send_cpu("A1", 0)
    #     layer.ForwardOutput.send_cpu("B1", 0)
    #     layer.BackwardWeight.send_cpu("A1", 0)
    #     layer.BackwardWeight.send_cpu("B1", 0)
    #
    # if sid == 0:
    #     layer.ForwardOutput.generate_cpu_tensor("A1", layer.ForwardOutput.get_cpu("A0").shape)
    #     layer.ForwardOutput.generate_cpu_tensor("B1", layer.ForwardOutput.get_cpu("B0").shape)
    #     layer.BackwardWeight.generate_cpu_tensor("A1", layer.BackwardWeight.get_cpu("A0").shape)
    #     layer.BackwardWeight.generate_cpu_tensor("B1", layer.BackwardWeight.get_cpu("B0").shape)
    #     layer.ForwardOutput.recv_cpu("A1", 1)
    #     layer.ForwardOutput.recv_cpu("B1", 1)
    #     layer.BackwardWeight.recv_cpu("A1", 1)
    #     layer.BackwardWeight.recv_cpu("B1", 1)
    #     layer_op_name = layer.LayerName + "Forward"
    #     InputQ, WeightQ = pre_quantize(input_f, weight_f, layer_op_name, 0)
    #
    #     OutputQ = conv2d_op(WeightQ, InputQ)
    #     OutputQ = mod_move_down(OutputQ)
    #
    #     output_f = post_quantize(OutputQ, layer_op_name, 0)
    #     # output_f = OutputQ
    #
    #     layer_op_name = layer.LayerName + "BackwardInput"
    #     OutputGradQ, WeightQ = pre_quantize(der_output_f, weight_f, layer_op_name, 0)
    #
    #     InputGradQ = conv2d_input_grad_op(WeightQ, OutputGradQ)
    #     InputGradQ = mod_move_down(InputGradQ)
    #
    #     input_grad_f = post_quantize(InputGradQ, layer_op_name, 0)
    #     # input_grad_f = InputGradQ
    #
    #     layer_op_name = layer.LayerName + "BackwardWeight"
    #     OutputGradQ, InputQ = pre_quantize(der_output_f, input_f, layer_op_name, 0)
    #
    #     WeightGradQ = conv2d_weight_grad_op(OutputGradQ, InputQ)
    #     WeightGradQ = mod_move_down(WeightGradQ)
    #
    #     weight_grad_f = post_quantize(WeightGradQ, layer_op_name, 0)
    #     # weight_grad_f = WeightGradQ
    #
    #     def recover_from_share(sharebase, open_share, close_share):
    #         # sharebase.transfer_enclave_to_cpu(name1)
    #         # sharebase.transfer_enclave_to_cpu(close_share)
    #         tensor_q = sharebase.get_cpu(open_share) + sharebase.get_cpu(close_share)
    #         return mod_move_down(tensor_q)
    #
    #     print("InputQ")
    #     enclave_input_q = recover_from_share(layer.ForwardOutput, "B0", "B1")
    #     compare_expected_actual(InputQ, enclave_input_q, get_relative=True, verbose=True)
    #     # print("InputQ:", InputQ[0, 0, 0, :])
    #     # print("enclave_input_q:", enclave_input_q[0, 0, 0, :])
    #     print("WeightQ")
    #     compare_expected_actual(WeightQ, recover_from_share(layer.ForwardOutput, "A0", "A1"), get_relative=True, verbose=True)
    #     print("OutputQ")
    #     layer.ForwardOutput.transfer_enclave_to_cpu("CQ")
    #     enclave_output_q = recover_from_share(layer.ForwardOutput, "CQ", "C1")
    #     compare_expected_actual(OutputQ, enclave_output_q, get_relative=True, verbose=True)
    #     # print("OutputQ:", OutputQ[0, 0, 0, :])
    #     # print("enclave_output_q:", enclave_output_q[0, 0, 0, :])
    #     print("Output")
    #     layer.transfer_enclave_to_cpu("output")
    #     compare_expected_actual(output_f, layer.get_cpu("output"), get_relative=True, verbose=True)
    #     # print("expected:", output_f[0, 0, 0, :])
    #     # print("actual:", layer.get_cpu("output")[0, 0, 0, :])
    #     print("OutputGradQ")
    #     compare_expected_actual(OutputGradQ, layer.BackwardInput.get_cpu("BQ"), get_relative=True, verbose=True)
    #     print("WeightQ")
    #     compare_expected_actual(WeightQ, layer.BackwardInput.get_cpu("AQ"), get_relative=True, verbose=True)
    #     print("InputGradQ")
    #     layer.BackwardInput.transfer_enclave_to_cpu("CQ")
    #     compare_expected_actual(InputGradQ, layer.BackwardInput.get_cpu("CQ"), get_relative=True, verbose=True)
    #     print("Input Grad")
    #     layer.transfer_enclave_to_cpu("DerInput")
    #     compare_expected_actual(input_grad_f, layer.get_cpu("DerInput"), get_relative=True, verbose=True)
    #     print("WeightGrad: InputQ")
    #     enclave_input_q = recover_from_share(layer.BackwardWeight, "B0", "B1")
    #     compare_expected_actual(InputQ, enclave_input_q, get_relative=True, verbose=True)
    #     print("WeightGrad: OutputGradQ")
    #     enclave_output_grad_q = recover_from_share(layer.BackwardWeight, "A0", "A1")
    #     compare_expected_actual(OutputGradQ, enclave_output_grad_q, get_relative=True, verbose=True)
    #     print("Weight Grad Q")
    #     layer.BackwardWeight.transfer_enclave_to_cpu("CQ")
    #     enclave_weight_grad_q = recover_from_share(layer.BackwardWeight, "CQ", "C1")
    #     compare_expected_actual(WeightGradQ, enclave_weight_grad_q, get_relative=True, verbose=True)
    #     print("Weight Grad")
    #     layer.transfer_enclave_to_cpu("DerWeight")
    #     compare_expected_actual(weight_grad_f, layer.get_cpu("DerWeight"), get_relative=True, verbose=True)

    dist.barrier()
    dist.destroy_process_group()


def local_shared_matmul(sid, master_addr, master_port, layer_name, layer_params):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size, n_input_features, n_output_features = layer_params
    print(f"batch_size, n_output_features: {batch_size, n_output_features, n_output_features}")
    x_shape = [batch_size, n_input_features]
    w_shape = [n_output_features, n_input_features]
    y_shape = [batch_size, n_output_features]

    Bf = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    der_output_f = generate_unquantized_tensor(SecretEnum.Error, y_shape).type(SecretConfig.dtypeForCpuOp)

    GlobalTensor.init()

    layer = SecretMatmulLayer(sid, layer_name, batch_size, n_output_features, n_input_features)
    layer.set_eid(GlobalTensor.get_eid())
    layer.init_shape()
    layer.link_tensors()
    layer.init(start_enclave=False)
    if sid != 2:
        layer.load_tensors(None, Bf, der_output_f, for_quantized=False)

    dist.barrier()
    layer.forward(need_quantize=True)
    dist.barrier()
    layer.backward(need_quantize=True)
    if sid != 2:
        layer.plain_forward(quantized_only=False)
        layer.plain_backward(quantized_only=False)
        layer.show_plain_error(quantized_only=False)

    dist.barrier()
    dist.destroy_process_group()


def local_plain_relu(sid, master_addr, master_port, layer_name, conv2d_params):
    print("===== Test: local_plain_relu =====")

    batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    print(f"batch_size, n_output_channel, n_input_channel, img_hw, filter_hw: "
          f"{batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    y_shape = x_shape

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    der_output_f = generate_unquantized_tensor(SecretEnum.Error, y_shape).type(SecretConfig.dtypeForCpuOp)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, layer_name + "Input", x_shape)
    target_layer = SecretReLULayer(sid, layer_name + "ReLU", True)

    layers = [input_layer, target_layer]

    input_layer.StoreInEnclave = False
    target_layer.register_prev_layer(input_layer)

    for layer in layers:
        layer.set_eid(GlobalTensor.get_eid())
        layer.init_shape()
        layer.link_tensors()
        layer.init(start_enclave=False)

    if sid != 2:
        target_layer.load_tensors(input_f, der_output_f)
        target_layer.forward()
        target_layer.backward()
        target_layer.plain_forward()
        target_layer.plain_backward()
        target_layer.show_plain_error()


def local_plain_maxpool2d(sid, master_addr, master_port, layer_name, conv2d_params):
    print("===== Test: local_plain_maxpool2d =====")

    batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    print(f"batch_size, n_output_channel, n_input_channel, img_hw, filter_hw: "
          f"{batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    filter_hw = 2
    output_hw = img_hw // filter_hw
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    y_shape = [batch_size, n_input_channel, output_hw, output_hw]

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    der_output_f = generate_unquantized_tensor(SecretEnum.Error, y_shape).type(SecretConfig.dtypeForCpuOp)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, layer_name + "Input", x_shape)
    target_layer = SecretMaxpool2dLayer(sid, layer_name + "Maxpool2d", filter_hw, True)

    layers = [input_layer, target_layer]

    input_layer.StoreInEnclave = True
    target_layer.register_prev_layer(input_layer)

    for layer in layers:
        layer.set_eid(GlobalTensor.get_eid())
        layer.init_shape()
        layer.link_tensors()
        layer.init(start_enclave=False)

    if sid != 2:
        target_layer.load_tensors(input_f, der_output_f)
        target_layer.forward()
        target_layer.backward()
        target_layer.plain_forward()
        target_layer.plain_backward()
        target_layer.show_plain_error()


def local_plain_batchnorm2d(sid, master_addr, master_port, layer_name, conv2d_params):
    print("===== Test: local_plain_batchnorm2d =====")

    batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    print(f"batch_size, n_output_channel, n_input_channel, img_hw, filter_hw: "
          f"{batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    # padding = 1
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    y_shape = x_shape

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    der_output_f = generate_unquantized_tensor(SecretEnum.Error, y_shape).type(SecretConfig.dtypeForCpuOp)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, layer_name + "Input", x_shape)
    target_layer = SecretBatchNorm2dLayer(sid, layer_name + "target")

    layers = [input_layer, target_layer]

    input_layer.StoreInEnclave = False
    target_layer.register_prev_layer(input_layer)

    for layer in layers:
        layer.set_eid(GlobalTensor.get_eid())
        layer.init_shape()
        layer.link_tensors()
        layer.init(start_enclave=False)

    if sid != 2:
        target_layer.load_tensors(input_f, der_output_f)
        n_iter = 4
        for i in range(n_iter):
            target_layer.forward()
            target_layer.backward()
            target_layer.plain_forward()
            target_layer.plain_backward()
            target_layer.show_plain_error()


def local_plain_flatten(sid, master_addr, master_port, layer_name, conv2d_params):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    print(f"batch_size, n_output_channel, n_input_channel, img_hw, filter_hw: "
          f"{batch_size, n_output_channel, n_input_channel, img_hw, filter_hw}")
    x_shape = [batch_size, n_input_channel, img_hw, img_hw]
    y_shape = x_shape

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    der_output_f = generate_unquantized_tensor(SecretEnum.Error, y_shape).type(SecretConfig.dtypeForCpuOp)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, layer_name + "Input", x_shape)
    target_layer = SecretFlattenLayer(sid, layer_name + "Flatten")

    layers = [input_layer, target_layer]

    input_layer.StoreInEnclave = False
    target_layer.register_prev_layer(input_layer)

    for layer in layers:
        layer.set_eid(GlobalTensor.get_eid())
        layer.init_shape()
        layer.link_tensors()
        layer.init(start_enclave=False)

    if sid != 2:
        target_layer.load_tensors(input_f, der_output_f)
        target_layer.forward()
        target_layer.backward()
        target_layer.plain_forward()
        target_layer.plain_backward()
        target_layer.show_plain_error()


def local_plain_output(sid, master_addr, master_port, layer_name, layer_params):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size, n_output_features = layer_params
    print(f"batch_size, n_output_features: "
          f"{batch_size, n_output_features}")
    x_shape = [batch_size, n_output_features]
    t_shape = [batch_size]

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    target_f = get_random_uniform(n_output_features, t_shape).type(torch.long)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, layer_name + "Input", x_shape)
    target_layer = SecretOutputLayer(sid, layer_name + "target")

    layers = [input_layer, target_layer]

    input_layer.StoreInEnclave = False
    target_layer.register_prev_layer(input_layer)

    for layer in layers:
        layer.set_eid(GlobalTensor.get_eid())
        layer.init_shape()
        layer.link_tensors()
        layer.init(start_enclave=False)

    if sid != 2:
        input_layer.set_input(input_f)
        target_layer.load_target(target_f)
        target_layer.forward()
        target_layer.backward()
        target_layer.plain_forward()
        target_layer.plain_backward()
        target_layer.show_plain_error()


def local_nn(sid, master_addr, master_port):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 16
    n_img_channel = 32
    img_hw = 32
    n_unit_fc1 = 64
    n_channel_conv1 = 16
    n_channel_conv2 = 16
    n_channel_conv3 = 32
    n_channel_conv4 = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]
    t_shape = [batch_size]

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    target_f = get_random_uniform(n_classes, t_shape).type(torch.long)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)
    conv1 = SecretConv2dLayer(sid, "Conv1", n_channel_conv1, 3)
    norm1 = SecretBatchNorm2dLayer(sid, "Norm1")
    relu1 = SecretReLULayer(sid, "Relu1")
    conv2 = SecretConv2dLayer(sid, "Conv2", n_channel_conv2, 3)
    norm2 = SecretBatchNorm2dLayer(sid, "Norm2")
    relu2 = SecretReLULayer(sid, "Relu2")
    pool2 = SecretMaxpool2dLayer(sid, "Pool2", 2)

    conv3 = SecretConv2dLayer(sid, "Conv3", n_channel_conv3, 3)
    norm3 = SecretBatchNorm2dLayer(sid, "Norm3")
    relu3 = SecretReLULayer(sid, "Relu3")
    conv4 = SecretConv2dLayer(sid, "Conv4", n_channel_conv4, 3)
    norm4 = SecretBatchNorm2dLayer(sid, "Norm4")
    relu4 = SecretReLULayer(sid, "Relu4")
    pool4 = SecretMaxpool2dLayer(sid, "Pool4", 2)

    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, n_unit_fc1)
    fc_norm1 = SecretBatchNorm1dLayer(sid, "FcNorm1")
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, n_classes)
    output_layer = SecretOutputLayer(sid, "OutputLayer")

    layers = [input_layer,
              conv1, norm1, relu1, conv2, norm2, relu2, pool2,
              conv3, norm3, relu3, conv4, norm4, relu4, pool4,
              flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    if sid != 2:
        input_layer.set_input(input_f)
        output_layer.load_target(target_f)

    dist.barrier()
    secret_nn.forward()
    secret_nn.backward()

    if sid != 2:
        secret_nn.plain_forward()
        secret_nn.plain_backward()
        secret_nn.show_plain_error()

        print("magnitude of input:", modest_magnitude(fc1.get_cpu("input")))
        print("magnitude of weight:", modest_magnitude(fc1.get_cpu("weight")))
        print("magnitude of DerOutput:", modest_magnitude(fc1.get_cpu("DerOutput")))


def local_sgd(sid, master_addr, master_port):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    NumEpoch = 2
    batch_size = 16
    n_img_channel = 32
    img_hw = 32
    n_unit_fc1 = 64
    n_channel_conv1 = 16
    n_channel_conv2 = 16
    n_channel_conv3 = 32
    n_channel_conv4 = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]
    t_shape = [batch_size]

    input_f = generate_unquantized_tensor(SecretEnum.Activate, x_shape).type(SecretConfig.dtypeForCpuOp)
    target_f = get_random_uniform(n_classes, t_shape).type(torch.long)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)
    conv1 = SecretConv2dLayer(sid, "Conv1", n_channel_conv1, 3)
    norm1 = SecretBatchNorm2dLayer(sid, "Norm1")
    relu1 = SecretReLULayer(sid, "Relu1")
    pool1 = SecretMaxpool2dLayer(sid, "Pool1", 2)

    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, n_unit_fc1)
    fc_norm1 = SecretBatchNorm1dLayer(sid, "FcNorm1")
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, n_classes)
    output_layer = SecretOutputLayer(sid, "OutputLayer")

    layers = [input_layer,
              conv1, norm1, relu1, pool1,
              flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    optim = SgdOptimizer(sid)
    optim.set_eid(GlobalTensor.get_eid())
    optim.set_layers(layers)

    if sid != 2:
        input_layer.set_input(input_f)
        output_layer.load_target(target_f)

    for epoch in range(NumEpoch):
        dist.barrier()
        secret_nn.forward()
        secret_nn.backward()

        if sid != 2:
            optim.update_params(test_with_ideal=True)

    print(f"SID: {sid} Finished")


def marshal_local_nd(sid, master_addr, master_port, local_func, shape):
    cnt_test = 0

    def give_test(shape_params):
        nonlocal sid, cnt_test
        cnt_test += 1
        layer_name = "SingleLayer%02d" % cnt_test
        print()
        print(f"sid: {sid}, params: {shape_params}")
        local_args = [layer_name, shape_params]
        marshal_process(sid, master_addr, master_port, local_func, local_args)

    give_test(shape)

def marshal_nonlinear(sid, master_addr, master_port, local_func):
    cnt_test = 0

    def give_test(shape_params):
        nonlocal sid, cnt_test
        cnt_test += 1
        layer_name = "SingleLayer%02d" % cnt_test
        print()
        print(f"sid: {sid}, params: {shape_params}")
        local_args = [layer_name, shape_params]
        marshal_process(sid, master_addr, master_port, local_func, local_args)

    batch_size = 2

    # batch_size, n_output_channel, n_input_channel, img_hw, filter_hw = conv2d_params
    give_test([128, 16, 32, 8, 3])
    give_test([batch_size, 64, 64, 32, 3])
    give_test([batch_size, 128, 128, 16, 3])
    give_test([batch_size, 256, 128, 8, 3])
    give_test([batch_size, 256, 256, 8, 3])
    give_test([batch_size, 256, 256, 4, 3])
    give_test([batch_size, 512, 512, 4, 3])
    give_test([batch_size, 512, 512, 2, 3])


def marshal_local_layer4d(sid, master_addr, master_port, local_func):
    def give_test(shape):
        marshal_local_nd(sid, master_addr, master_port, local_func, shape)

    batch_size = 512

    give_test([128, 16, 32, 8, 3])
    give_test([batch_size, 64, 3, 32, 3])
    give_test([batch_size, 128, 64, 16, 3])
    # give_test([batch_size, 128, 128, 16, 3])
    give_test([batch_size, 256, 128, 8, 3])
    give_test([batch_size, 256, 256, 8, 3])
    give_test([batch_size, 512, 256, 4, 3])
    give_test([batch_size, 512, 512, 4, 3])
    give_test([batch_size, 512, 512, 2, 3])


def argparser_distributed():
    default_sid = -1
    default_ip = "127.0.0.1"
    default_port = "29501"
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", "-s",
                        type=int,
                        default=default_sid,
                        help="The ID of the server")
    parser.add_argument("--ip",
                        dest="MasterAddr",
                        default=default_ip,
                        help="The Master Address for communication")
    parser.add_argument("--port",
                        dest="MasterPort",
                        default=default_port,
                        help="The Master Port for communication")
    parser.add_argument("--test",
                        dest="TestToRun",
                        default="all",
                        help="The Test to run")
    args = parser.parse_args()
    input_sid = args.sid
    MasterAddr = args.MasterAddr
    MasterPort = args.MasterPort
    test_to_run = args.TestToRun

    return input_sid, MasterAddr, MasterPort, test_to_run


def seed_torch(seed=123):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    input_sid, MasterAddr, MasterPort, test_to_run = argparser_distributed()
    print("input_sid, MasterAddr, MasterPort:", input_sid, MasterAddr, MasterPort)

    sys.stdout = Logger()
    print("====== New Tests ======")

    seed_torch(125)

    if test_to_run == "conv2d_quantized" or test_to_run == "all":
        marshal_local_layer4d(input_sid, MasterAddr, MasterPort, local_shared_conv2d_quantized)
    if test_to_run == "conv2d" or test_to_run == "all":
        marshal_local_layer4d(input_sid, MasterAddr, MasterPort, local_shared_conv2d)
    if test_to_run == "matmul" or test_to_run == "all":
        marshal_local_nd(input_sid, MasterAddr, MasterPort, local_shared_matmul, [4096, 4096, 4096])
    if test_to_run == "relu" or test_to_run == "all":
        marshal_nonlinear(input_sid, MasterAddr, MasterPort, local_plain_relu)
    if test_to_run == "maxpooling" or test_to_run == "all":
        marshal_nonlinear(input_sid, MasterAddr, MasterPort, local_plain_maxpool2d)
    if test_to_run == "flatten" or test_to_run == "all":
        marshal_local_nd(input_sid, MasterAddr, MasterPort, local_plain_flatten, [64, 16, 32, 8, 3])
    if test_to_run == "bn2d" or test_to_run == "all":
        marshal_nonlinear(input_sid, MasterAddr, MasterPort, local_plain_batchnorm2d)
    if test_to_run == "output" or test_to_run == "all":
        marshal_local_nd(input_sid, MasterAddr, MasterPort, local_plain_output, [32 * 32, 10])
    if test_to_run == "nn" or test_to_run == "all":
        marshal_process(input_sid, MasterAddr, MasterPort, local_nn, [])
    if test_to_run == "sgd" or test_to_run == "all":
        marshal_process(input_sid, MasterAddr, MasterPort, local_sgd, [])
