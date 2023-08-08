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
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.linear_shares import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.logger_utils import Logger
from python.quantize_net import NetQ
from python.test_linear_shares import argparser_distributed, marshal_process, load_cifar10, seed_torch
from python.timer_utils import NamedTimerInstance, VerboseLevel, NamedTimer
from python.torch_utils import compare_expected_actual

device_cuda = torch.device("cuda:0")

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


def local_vgg9(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()
    
    batch_size = 2
    n_img_channel = 3
    img_hw = 32
    n_classes = 10

    n_unit_fc1 = 512
    n_unit_fc2 = 512

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=32)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)

    def generate_conv_module(index, n_channel_conv, is_big=True):
        res = []
        if is_big:
            conv_local_1 = SecretConv2dLayer(sid, f"Conv{index}A", n_channel_conv, 3)
            norm_local_1 = SecretBatchNorm2dLayer(sid, f"Norm1{index}A")
            relu_local_1 = SecretReLULayer(sid, f"Relu{index}A")
            res += [conv_local_1, norm_local_1, relu_local_1]
        conv_local_2 = SecretConv2dLayer(sid, f"Conv{index}B", n_channel_conv, 3)
        norm_local_2 = SecretBatchNorm2dLayer(sid, f"Norm{index}B")
        relu_local_2 = SecretReLULayer(sid, f"Relu{index}B")
        pool_local_2 = SecretMaxpool2dLayer(sid, f"Pool{index}B", 2)
        res += [conv_local_2, norm_local_2, relu_local_2, pool_local_2]
        return res


    conv_module_1 = generate_conv_module(1, 64, is_big=False)
    conv_module_2 = generate_conv_module(2, 128, is_big=False)
    conv_module_3 = generate_conv_module(3, 256, is_big=True)
    conv_module_4 = generate_conv_module(4, 512, is_big=True)
    conv_module_5 = generate_conv_module(5, 512, is_big=True)
    all_conv_module = conv_module_1 + conv_module_2 + conv_module_3 + conv_module_4 + conv_module_5
    # all_conv_module = conv_module_1 + conv_module_2

    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, n_unit_fc1)
    # fc_norm1 = SecretBatchNorm1dLayer(sid, "FcNorm1")
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, n_unit_fc2)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, n_classes)
    output_layer = SecretOutputLayer(sid, "OutputLayer")

    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer] + all_conv_module + [flatten, fc1, fc_relu1, fc2, fc_relu2, fc3, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    secret_optim = SgdOptimizer(sid)
    secret_optim.set_eid(GlobalTensor.get_eid())
    secret_optim.set_layers(layers)

    input_layer.StoreInEnclave = False

    if is_compare:
        net = NetQ()
        net.to(device_cuda)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    conv1, norm1, relu1, pool1 = conv_module_1[0:4]
    conv2, norm2, relu2, pool2 = conv_module_2[0:4]
    conv3, norm3 = conv_module_3[0:2]
    conv4, norm4 = conv_module_3[3:5]
    conv5, norm5 = conv_module_4[0:2]
    conv6, norm6 = conv_module_4[3:5]
    conv7, norm7 = conv_module_5[0:2]
    conv8, norm8, relu8, pool8 = conv_module_5[3:7]


    validation_net = NetQ()
    validation_net.to(torch.device("cuda:0"))

    def validation_accuracy():
        correct = 0
        total = 0
        conv1.inject_to_plain(validation_net.conv1)
        norm1.inject_to_plain(validation_net.norm1)
        conv2.inject_to_plain(validation_net.conv2)
        norm2.inject_to_plain(validation_net.norm2)
        conv3.inject_to_plain(validation_net.conv3)
        norm3.inject_to_plain(validation_net.norm3)
        conv4.inject_to_plain(validation_net.conv4)
        norm4.inject_to_plain(validation_net.norm4)
        conv5.inject_to_plain(validation_net.conv5)
        norm5.inject_to_plain(validation_net.norm5)
        conv6.inject_to_plain(validation_net.conv6)
        norm6.inject_to_plain(validation_net.norm6)
        conv7.inject_to_plain(validation_net.conv7)
        norm7.inject_to_plain(validation_net.norm7)
        conv8.inject_to_plain(validation_net.conv8)
        norm8.inject_to_plain(validation_net.norm8)
        fc1.inject_to_plain(validation_net.fc1)
        fc2.inject_to_plain(validation_net.fc2)
        fc3.inject_to_plain(validation_net.fc3)
        validation_net.to(device_cuda)

        for data in testloader:
            images, labels = data[0].to(device_cuda), data[1].to(device_cuda)
            outputs = validation_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        torch.cuda.empty_cache()

    def sync_layer_params():
        conv1.inject_params(net.conv1)
        norm1.inject_params(net.norm1)
        conv2.inject_params(net.conv2)
        norm2.inject_params(net.norm2)
        conv3.inject_params(net.conv3)
        norm3.inject_params(net.norm3)
        conv4.inject_params(net.conv4)
        norm4.inject_params(net.norm4)
        conv5.inject_params(net.conv5)
        norm5.inject_params(net.norm5)
        conv6.inject_params(net.conv6)
        norm6.inject_params(net.norm6)
        conv7.inject_params(net.conv7)
        norm7.inject_params(net.norm7)
        conv8.inject_params(net.conv8)
        norm8.inject_params(net.norm8)
        fc1.inject_params(net.fc1)
        fc2.inject_params(net.fc2)
        fc3.inject_params(net.fc3)

        register_layer(net.relu2, "relu2")
        register_layer(net.pool2, "pool2")
        register_layer(net.relu8, "relu8")
        register_layer(net.pool8, "pool8")

        register_weight_layer(net.conv1, "conv1")
        register_weight_layer(net.norm1, "norm1")
        register_weight_layer(net.conv2, "conv2")
        register_weight_layer(net.norm2, "norm2")
        register_weight_layer(net.conv3, "conv3")
        register_weight_layer(net.conv4, "conv4")
        register_weight_layer(net.conv5, "conv5")
        register_weight_layer(net.conv6, "conv6")
        register_weight_layer(net.conv7, "conv7")
        register_weight_layer(net.conv8, "conv8")
        register_weight_layer(net.norm8, "norm8")
        register_weight_layer(net.fc1, "fc1")
        register_weight_layer(net.fc2, "fc2")
        register_weight_layer(net.fc3, "fc3")

    if sid != 2 and is_compare:
        sync_layer_params()

    def compare_layer_weight(layer: SecretLinearLayerBase, layer_name: str) -> None:
        print(f"S{sid}: Compare with netQ layer:", layer_name)
        layer.make_sure_cpu_is_latest("weight")
        compare_expected_actual(get_layer_weight(layer_name), layer.get_cpu("weight"), get_relative=True, verbose=True)

    if sid == 0 and is_compare:
        compare_layer_weight(conv1, "conv1")
        compare_layer_weight(conv2, "conv2")
        compare_layer_weight(conv3, "conv3")
        compare_layer_weight(conv4, "conv4")
        compare_layer_weight(conv5, "conv5")
        compare_layer_weight(conv6, "conv6")
        compare_layer_weight(conv7, "conv7")
        compare_layer_weight(conv8, "conv8")
        compare_layer_weight(fc1, "fc1")
        compare_layer_weight(fc2, "fc2")
        compare_layer_weight(fc3, "fc3")

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 200
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            if train_counter % 30 == 0 and sid == 0:
                validation_accuracy()

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                def compute_expected_nn():
                    optimizer.zero_grad()
                    outputs = net(input_f.to(device_cuda))
                    loss = criterion(outputs, target_f.to(device_cuda))
                    loss.backward()
                    optimizer.step()
                    print("netQ loss:", loss)

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                secret_nn.forward()
                
                secret_nn.backward()

                if sid == 0 and is_compare:
                    compute_expected_nn()
                    compare_weight_layer(conv1, "conv1")
                    compare_weight_layer(norm1, "norm1")
                    compare_weight_layer(conv2, "conv2")
                    compare_weight_layer(norm2, "norm2")
                    compare_layer(relu2, "relu2")
                    compare_layer(pool2, "pool2")
                    compare_weight_layer(conv3, "conv3")
                    compare_weight_layer(conv4, "conv4")
                    compare_weight_layer(conv5, "conv5")
                    compare_weight_layer(conv6, "conv6")
                    compare_weight_layer(conv7, "conv7")
                    compare_weight_layer(conv8, "conv8")
                    compare_weight_layer(norm8, "norm8")
                    compare_layer(relu8, "relu8")
                    compare_layer(pool8, "pool8")
                    compare_weight_layer(fc1, "fc1")
                    compare_weight_layer(fc2, "fc2")
                    compare_weight_layer(fc3, "fc3")

                if sid != 2:
                    if is_compare:
                        # secret_nn.plain_forward()
                        # secret_nn.plain_backward()
                        # secret_nn.show_plain_error()
                        # secret_optim.update_params(test_with_ideal=True)
                        secret_optim.update_params(test_with_ideal=False)
                        pass
                    else:
                        secret_optim.update_params(test_with_ideal=False)
                        pass

                    running_loss += secret_nn.get_loss()
                    print(f"TrainCounter: {train_counter}, current_loss: {secret_nn.get_loss()}")

                with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()
        NamedTimer.end("TrainValidationEpoch")


if __name__ == "__main__":
    input_sid, MasterAddr, MasterPort, test = argparser_distributed()

    sys.stdout = Logger()
    print("====== New Tests ======")
    print("input_sid, MasterAddr, MasterPort", input_sid, MasterAddr, MasterPort)

    seed_torch(123)

    marshal_process(input_sid, MasterAddr, MasterPort, local_vgg9, [])
