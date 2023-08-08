from __future__ import print_function

import sys
import time
from collections import namedtuple
import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter
from pdb import set_trace as st

from python.logger_utils import Logger
from python.enclave_interfaces import GlobalState
from python.common_torch import mod_move_down
from python.data import get_data, num_classes_dict


dtype = torch.float
device = torch.device("cuda:0")

minibatch = 512
# n_classes = 10  # Create random Tensors to hold inputs and outputs


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


NamedParam = namedtuple("NamedParam", ("Name", "Param"))
store_configs = {}

def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)

def my_quantize_tensor(data, bits=8, mode="stochastic"):
    ebit = 8
    max_entry = torch.max(torch.abs(data)).item()
    if max_entry == 0: return data
    max_exponent = math.floor(math.log2(max_entry))
    max_exponent = min(max(max_exponent, -2 ** (ebit - 1)), 2 ** (ebit - 1) - 1)
    i = data * 2 ** (-max_exponent + (bits - 2))
    if mode == "stochastic":
        add_r_(i)
        i.floor_()
    elif mode == "nearest":
        i.round_()
    i.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1) # -2 ** (bits-1) to 2 ** (bits-1) - 1
    temp = i + 2 ** (bits - 1) 
    temp = temp.byte()
    return temp, max_exponent, bits

def my_dequantize_tensor(data, max_exponent, bits):
    data = data.float()
    data -= 2 ** (bits - 1) 
    return data * 2 ** (max_exponent - (bits - 2))


def swalp_quantize(param, bits=8, mode="stochastic"):
    data = param.Param
    ebit = 8
    max_entry = torch.max(torch.abs(data)).item()
    if max_entry == 0: return data
    max_exponent = math.floor(math.log2(max_entry))
    max_exponent = min(max(max_exponent, -2 ** (ebit - 1)), 2 ** (ebit - 1) - 1)
    i = data * 2 ** (-max_exponent + (bits - 2))
    if mode == "stochastic":
        add_r_(i)
        i.floor_()
    elif mode == "nearest":
        i.round_()
    i.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    temp = i
    store_configs[param.Name] = (max_exponent, bits)
    return temp


def quantize_op(param, layer_op_name):
    return swalp_quantize(param, 8)


def dequantize_op(param, layer_op_name):
    x_exp, x_bits = store_configs[layer_op_name + "X"]
    y_exp, y_bits = store_configs[layer_op_name + "Y"]
    return param.Param * 2 ** (x_exp - (x_bits - 2) + y_exp - (y_bits - 2))


def pre_quantize(param_x, param_y, layer_op_name, iter_time):
    named_x = NamedParam(layer_op_name + "X", param_x)
    named_y = NamedParam(layer_op_name + "Y", param_y)

    x_q = quantize_op(named_x, layer_op_name)
    y_q = quantize_op(named_y, layer_op_name)

    return x_q, y_q


def post_quantize(param_z, layer_op_name, iter_time):
    # return param_z
    named_z = NamedParam(layer_op_name + "ZQ", param_z)
    z_f = dequantize_op(named_z, layer_op_name)
    return z_f


def string_to_tensor(s):
    return torch.tensor(np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8))


def tensor_to_string(t):
    return "".join(map(chr, t))


# Inherit from Function
class QuantizeMatMulFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, layer_name=None, iter_time=0):

        layer_op_name = layer_name + "Forward"
        InputQ, WeightQ = pre_quantize(input, weight, layer_op_name, iter_time)

        OutputQ = InputQ.mm(WeightQ.t())
        OutputQ = mod_move_down(OutputQ)

        output = post_quantize(OutputQ, layer_op_name, iter_time)

        ctx.save_for_backward(input, weight, bias, string_to_tensor(layer_name), torch.tensor(iter_time))

        return output

    @staticmethod
    def backward(ctx, grad_output, ):
        input, weight, bias, layer_name, iter_time = ctx.saved_tensors
        layer_name = tensor_to_string(layer_name)
        iter_time = int(iter_time)
        input_grad = weight_grad = bias_grad = None

        if ctx.needs_input_grad[0]:
            layer_op_name = layer_name + "BackwardInput"
            OutputGradQ, WeightQ = pre_quantize(grad_output, weight, layer_op_name, iter_time)

            InputGradQ = OutputGradQ.mm(WeightQ)
            InputGradQ = mod_move_down(InputGradQ)

            input_grad = post_quantize(InputGradQ, layer_op_name, iter_time)

        if ctx.needs_input_grad[1]:
            layer_op_name = layer_name + "BackwardWeight"
            OutputGradQ, InputQ = pre_quantize(grad_output, input, layer_op_name, iter_time)

            WeightGradQ = OutputGradQ.t().mm(InputQ)
            WeightGradQ = mod_move_down(WeightGradQ)

            weight_grad = post_quantize(WeightGradQ, layer_op_name, iter_time)

        return input_grad, weight_grad, bias_grad, None, None


class QuantizeMatmul(nn.Module):
    def __init__(self, input_features, output_features, bias=None, layer_name=None):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layer_name = layer_name

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features).type(dtype).to(device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features).type(dtype).to(device))
        else:
            self.register_parameter('bias', None)

        self.iter_time = -1

        # Not a very smart way to initialize weights
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        self.iter_time += 1
        return QuantizeMatMulFunction.apply(input, self.weight, self.bias, self.layer_name, self.iter_time)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, layer_name={}, iter_time={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.layer_name, self.iter_time
        )


# Inherit from Function
class QuantizeConv2dFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, layer_name=None, iter_time=0):

        layer_op_name = layer_name + "Forward"
        InputQ, WeightQ = pre_quantize(input, weight, layer_op_name, iter_time)

        OutputQ = F.conv2d(InputQ, WeightQ, padding=1)
        OutputQ = mod_move_down(OutputQ)

        output = post_quantize(OutputQ, layer_op_name, iter_time)

        ctx.save_for_backward(input, weight, bias, string_to_tensor(layer_name), torch.tensor(iter_time))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, layer_name, iter_time = ctx.saved_tensors
        layer_name = tensor_to_string(layer_name)
        iter_time = int(iter_time)
        input_grad = weight_grad = bias_grad = None

        if ctx.needs_input_grad[0]:
            layer_op_name = layer_name + "BackwardInput"
            OutputGradQ, WeightQ = pre_quantize(grad_output, weight, layer_op_name, iter_time)

            InputGradQ = F.conv_transpose2d(OutputGradQ, WeightQ, padding=1)
            InputGradQ = mod_move_down(InputGradQ)

            input_grad = post_quantize(InputGradQ, layer_op_name, iter_time)

        if ctx.needs_input_grad[1]:
            layer_op_name = layer_name + "BackwardWeight"
            OutputGradQ, InputQ = pre_quantize(grad_output, input, layer_op_name, iter_time)

            WeightGradQ = torch.transpose(
                F.conv2d(torch.transpose(InputQ, 0, 1),
                         torch.transpose(OutputGradQ, 0, 1),
                         padding=1),
                0, 1)
            WeightGradQ = mod_move_down(WeightGradQ)

            weight_grad = post_quantize(WeightGradQ, layer_op_name, iter_time)

        return input_grad, weight_grad, bias_grad, None, None


class QuantizeConv2d(nn.Module):
    def __init__(self, NumInputChannel, NumOutputChannel, FilterHW, padding=1, bias=None, layer_name=None):
        super().__init__()
        self.NumInputChannel = NumInputChannel
        self.NumOutputChannel = NumOutputChannel
        self.FilterHW = FilterHW
        self.layer_name = layer_name

        self.weight = nn.Parameter(
            torch.Tensor(self.NumOutputChannel, self.NumInputChannel, self.FilterHW, self.FilterHW)
                .type(dtype).to(device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.NumOutputChannel).type(dtype).to(device))
        else:
            self.register_parameter('bias', None)

        self.iter_time = -1

        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        self.iter_time += 1
        return QuantizeConv2dFunction.apply(input, self.weight, self.bias, self.layer_name, self.iter_time)

    def extra_repr(self):
        return 'NumInputChannel={}, NumOutputChannel={}, FilterHW={}, bias={}, layer_name={}, iter_time={}'.format(
            self.NumInputChannel, self.NumOutputChannel, self.FilterHW, self.bias is not None,
            self.layer_name, self.iter_time
        )


class CTX(object):
    saved_tensors = None

    def __init__(self):
        pass

    def save_for_backward(self, *args):
        self.saved_tensors = args


ctx = CTX()


class NetQVgg16(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1_1 = QuantizeConv2d(3, 64, 3, layer_name="conv1_1")
        self.norm1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = QuantizeConv2d(64, 64, 3, layer_name="conv1_2")
        self.norm1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = QuantizeConv2d(64, 128, 3, layer_name="conv2_1")
        self.norm2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = QuantizeConv2d(128, 128, 3, layer_name="conv2_2")
        self.norm2_2 = nn.BatchNorm2d(128)
        self.NumElemFlatten = 8192

        self.conv3_1 = QuantizeConv2d(128, 256, 3, layer_name="conv3_1")
        self.norm3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = QuantizeConv2d(256, 256, 3, layer_name="conv3_2")
        self.norm3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = QuantizeConv2d(256, 256, 3, layer_name="conv3_3")
        self.norm3_3 = nn.BatchNorm2d(256)

        self.conv4_1 = QuantizeConv2d(256, 512, 3, layer_name="conv4_1")
        self.norm4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = QuantizeConv2d(512, 512, 3, layer_name="conv4_2")
        self.norm4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = QuantizeConv2d(512, 512, 3, layer_name="conv4_3")
        self.norm4_3 = nn.BatchNorm2d(512)

        self.conv5_1 = QuantizeConv2d(512, 512, 3, layer_name="conv5_1")
        self.norm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = QuantizeConv2d(512, 512, 3, layer_name="conv5_2")
        self.norm5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = QuantizeConv2d(512, 512, 3, layer_name="conv5_3")
        self.norm5_3 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        self.normfc1 = nn.BatchNorm1d(512)
        self.normfc2 = nn.BatchNorm1d(512)
        self.normfc3 = nn.BatchNorm1d(10)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 512, layer_name="fc1")
        self.fc2 = QuantizeMatmul(512, 512, layer_name="fc2")
        self.fc3 = QuantizeMatmul(512, n_class, layer_name="fc3")

    def forward_without_bn(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_bn(self, x):
        x = F.relu(self.norm1_1(self.conv1_1(x)))
        x = F.relu(self.norm1_2(self.conv1_2(x)))
        x = self.pool(x)

        x = F.relu(self.norm2_1(self.conv2_1(x)))
        x = F.relu(self.norm2_2(self.conv2_2(x)))
        x = self.pool(x)

        x = F.relu(self.norm3_1(self.conv3_1(x)))
        x = F.relu(self.norm3_2(self.conv3_2(x)))
        x = F.relu(self.norm3_3(self.conv3_3(x)))
        x = self.pool(x)

        x = F.relu(self.norm4_1(self.conv4_1(x)))
        x = F.relu(self.norm4_2(self.conv4_2(x)))
        x = F.relu(self.norm4_3(self.conv4_3(x)))
        x = self.pool(x)

        x = F.relu(self.norm5_1(self.conv5_1(x)))
        x = F.relu(self.norm5_2(self.conv5_2(x)))
        x = F.relu(self.norm5_3(self.conv5_3(x)))
        x = self.pool(x)

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)
        # return self.forward_with_bn(x)


class NetQ(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = QuantizeConv2d(64, 128, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 8192

        self.conv3 = QuantizeConv2d(128, 256, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = QuantizeConv2d(256, 256, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = QuantizeConv2d(256, 512, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(512)
        self.conv6 = QuantizeConv2d(512, 512, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = QuantizeConv2d(512, 512, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(512)
        self.conv8 = QuantizeConv2d(512, 512, 3, layer_name="conv8")
        self.norm8 = nn.BatchNorm2d(512)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        # self.normfc1 = nn.BatchNorm1d(512)
        # self.normfc2 = nn.BatchNorm1d(512)
        # self.normfc3 = nn.BatchNorm1d(n_classes)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 512, layer_name="fc1")
        self.fc2 = QuantizeMatmul(512, 512, layer_name="fc2")
        self.fc3 = QuantizeMatmul(512, n_class, layer_name="fc3")

    def forward_without_bn(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(self.relu2((self.conv2(x))))

        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_bn(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.norm2(self.conv2(x))))

        x = self.pool(F.relu(self.norm4(self.conv4(F.relu(self.norm3(self.conv3(x)))))))
        x = self.pool(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(x)))))))
        x = self.pool8(F.relu(self.norm8(self.conv8(F.relu(self.norm7(self.conv7(x)))))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        # return self.forward_without_bn(x)
        return self.forward_with_bn(x)

def main():
    sys.stdout = Logger()
    writer = SummaryWriter('log')
    running_name = "stochastic"

    print("CUDA is available:", torch.cuda.is_available())
    # torch.backends.cudnn.deterministic = True

    dataset_name = "IDC"
    data_path = "./data"
    num_workers = 2
    n_class = num_classes_dict[dataset_name]

    loaders = get_data(dataset_name, data_path, minibatch, num_workers)
    trainloader = loaders["train"]
    testloader = loaders["test"]

    net = NetQ(n_class)
    # net = NetQVgg16(n_class)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # register_linear_layer(net.conv2, "conv2")

    NumShowInter = 60
    NumEpoch = 200
    IterCounter = -1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    training_start = time.time()
    train_total, train_correct = 0, 0
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        GlobalState.set_iter_epoch(epoch)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            IterCounter += 1
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if IterCounter % NumShowInter == NumShowInter - 1 or (epoch == 0 and i == 0):  # print every 2000 mini-batches
                elapsed_time = time.time() - training_start
                print('[%d, %5d, %6d, %6f] loss: %.3f, train acc: %5f' %
                      (epoch + 1, i + 1, IterCounter, elapsed_time, running_loss / NumShowInter,
                       train_correct / train_total * 100))
                writer.add_scalar(f"Train/{running_name}/Loss", running_loss / NumShowInter, IterCounter)

                running_loss = 0.00
                correct = 0
                total = 0

                # store_layer_name = 'conv2'
                # store_name = f"quantize_{store_layer_name}_{epoch + 1}_{i + 1}"
                # store_layer(store_layer_name, store_name)

                with torch.no_grad():
                    for data in testloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d test images: %6f %%' % (total, 100 * correct / total))
                    writer.add_scalar(f'Test/{running_name}/Accu', correct / total, IterCounter)

        scheduler.step()

    print('Finished Training')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
