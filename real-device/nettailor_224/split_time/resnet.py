"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from pdb import set_trace as st
import copy
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

def conv3x3(in_planes, out_planes, stride=1, use_bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=use_bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.in_channels = inplanes
        self.out_channels = planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * self.in_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        if self.downsample is not None:
            flops += spatial_dim * self.in_channels * self.out_channels * 1
            flops += spatial_dim * self.out_channels * 2
        return flops

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, img_size=224, ratio=0.5):
        self.inplanes = int(64*ratio)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, int(64*ratio), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*ratio))
        self.relu = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64*ratio), layers[0])
        self.layer2 = self._make_layer(block, int(128*ratio), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*ratio), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*ratio), layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*ratio) * block.expansion, num_classes)
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.layer_config = layers
        self.img_size = img_size
        self.num_classes = num_classes
        self.ratio = ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()
        self.collect_to_sequential()

    def init_layer_config(self):
        self.res_blocks = sum(self.layer_config)
        self.total_blocks = self.res_blocks + 2
        self.backward_blocks = ['fc']
        self.forward_blocks = ['conv1']
        layers_and_idxs = [(i+1, layer) for i, layer in enumerate(self.layer_config)]
        layers_and_idxs_reverse = copy.deepcopy(layers_and_idxs)
        layers_and_idxs_reverse.reverse()

        for layer_idx, layers in layers_and_idxs:
            for block_idx in range(layers):
                name = f"layer{layer_idx}.{block_idx}"
                self.forward_blocks.append(name)
        self.forward_blocks.append('fc')
        self.block_names = copy.deepcopy(self.forward_blocks)
        self.forward_blocks.append('end')
        # print("Forward blocks: ", self.forward_blocks)

        for layer_idx, layers in layers_and_idxs_reverse:
            for block_idx in range(layers, 0, -1):
                name = f"layer{layer_idx}.{block_idx-1}"
                self.backward_blocks.append(name)
        # self.backward_blocks.append('bn1')
        self.backward_blocks.append('conv1')
        self.backward_blocks.append('start')
        # print("Backward blocks: ", self.backward_blocks)

        self.parameter_names = []
        for name, _ in self.named_parameters():
            self.parameter_names.append(name)
        self.reverse_parameter_names = copy.deepcopy(self.parameter_names)
        self.reverse_parameter_names.reverse()
        # print("Forward parameters: ", self.parameter_names)
        # print("Backward parameters: ", self.reverse_parameter_names)

    def collect_to_sequential(self):
        block0 = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
        )
        blocks = [block0]
        for i in range(len(self.layer1)):
            block = self.layer1[i]
            blocks.append(block)
        for i in range(len(self.layer2)):
            block = self.layer2[i]
            blocks.append(block)
        for i in range(len(self.layer3)):
            block = self.layer3[i]
            blocks.append(block)
        for i in range(len(self.layer4)):
            block = self.layer4[i]
            blocks.append(block)

        block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.num_classes)
        )
        blocks.append(block)
        self.block_layers = nn.Sequential(
            *blocks
        )
        self.num_blocks = len(self.block_layers)

        self.block_input_shapes = [
            (3, 224, 224),
            (64, 56, 56),
        ]
        for i in range(len(self.layer1)):
            self.block_input_shapes.append(
                (64, 56, 56)
            )
        for i in range(len(self.layer2)):
            self.block_input_shapes.append(
                (128, 28, 28)
            )
        for i in range(len(self.layer3)):
            self.block_input_shapes.append(
                (256, 14, 14)
            )
        for i in range(len(self.layer4)):
            self.block_input_shapes.append(
                (512, 7, 7)
            )


    def block_forward_inference_time(self, batch_size=1, iteration=5):
        assert self.ratio == 1
        mean_times, mean_throughputs = [], []
        error_times, error_throughputs = [], []
        for block_idx in range(self.num_blocks):
            forward_blocks = self.block_layers[:block_idx+1]

            input_shape = (batch_size, 3, 224, 224)
            input = torch.rand(input_shape) 
            forward_blocks(input)

            inference_times = []
            throughputs = []
            
            for i in range(iteration):
                start = time.time()
                forward_blocks(input)
                end = time.time()
                elapse = (end-start) * 1000 / batch_size
                inference_times.append(elapse)
                throughputs.append(1000 / inference_times[-1])

            inf_mean = round(np.mean(inference_times), 2)
            inf_std = round(np.std(inference_times), 2)
            th_mean = round(np.mean(throughputs), 2)
            th_std = round(np.std(throughputs), 2)

            mean_times.append(inf_mean)
            mean_throughputs.append(th_mean)
            error_times.append(inf_std)
            error_throughputs.append(th_std)

            # print(f"Forward block {block_idx+1} Time mean {inf_mean:.4f} ms, std {inf_std:.4f} ms. Throughput mean {th_mean:.4f}, std {th_std:.4f}")

        forward_params = list(self.forward_block_params.values())[1:]
        forward_flops = list(self.forward_block_flops.values())[1:]

        print(f"resnet18_block_forward_params = ", forward_params)
        print(f"resnet18_block_forward_flops = ", forward_flops)

        print(f"resnet18_block_forward_time_mean = ", mean_times)
        print(f"resnet18_block_forward_time_error = ", error_times)
        print(f"resnet18_block_forward_throughput_mean = ", mean_throughputs)
        print(f"resnet18_block_forward_throughput_error = ", error_throughputs)

    def block_backward_inference_time(self, batch_size=1, iteration=5):
        assert self.ratio == 1
        mean_times, mean_throughputs = [], []
        error_times, error_throughputs = [], []
        
        for block_idx in range(self.num_blocks):
            backward_blocks = self.block_layers[-(block_idx+1):]

            input_shape = (batch_size,) + self.block_input_shapes[-(block_idx+1)]
            input = torch.rand(input_shape) 
            backward_blocks(input)

            inference_times = []
            throughputs = []
            
            for i in range(iteration):
                start = time.time()
                backward_blocks(input)
                end = time.time()
                elapse = (end-start) * 1000 / batch_size
                inference_times.append(elapse)
                throughputs.append(1000 / inference_times[-1])

            inf_mean = round(np.mean(inference_times), 2)
            inf_std = round(np.std(inference_times), 2)
            th_mean = round(np.mean(throughputs), 2)
            th_std = round(np.std(throughputs), 2)

            mean_times.append(inf_mean)
            mean_throughputs.append(th_mean)
            error_times.append(inf_std)
            error_throughputs.append(th_std)

            # print(f"Backward block {block_idx+1} Time mean {inf_mean:.4f} ms, std {inf_std:.4f} ms. Throughput mean {th_mean:.4f}, std {th_std:.4f}")

        backward_params = list(self.backward_block_params.values())[1:]
        backward_flops = list(self.backward_block_flops.values())[1:]

        print(f"resnet18_block_backward_params = ", backward_params)
        print(f"resnet18_block_backward_flops = ", backward_flops)

        print(f"resnet18_block_backward_time_mean = ", mean_times)
        print(f"resnet18_block_backward_time_error = ", error_times)
        print(f"resnet18_block_backward_throughput_mean = ", mean_throughputs)
        print(f"resnet18_block_backward_throughput_error = ", error_throughputs)

    def _make_layer(self, block, planes, blocks, stride=1, dropout=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ends = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # for layer in self.layer1:
        #     x = layer(x)
        x = self.layer1(x)

        # for layer in self.layer2:
        #     x = layer(x)
        x = self.layer2(x)

        # for layer in self.layer3:
        #     x = layer(x)
        x = self.layer3(x)

        # for layer in self.layer4:
        #     x = layer(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
    
    def load_pretrained(self, fn):
        checkpoint = torch.load(fn)
        state_dict = self.state_dict()
        for k in state_dict.keys():
            state_dict[k] = checkpoint['state_dict'][k]
        self.load_state_dict(state_dict)


    def set_deep_layers(self, num_layers, pretrained):
        assert num_layers <= self.total_blocks

        bar_layer_name = self.backward_blocks[num_layers]
        update_param_names, remain_param_names = [], []
        get_to_bar = False
        for name in self.reverse_parameter_names:
            if name.startswith(bar_layer_name):
                get_to_bar = True
            if not get_to_bar:
                update_param_names.append(name)
            else:
                remain_param_names.append(name)
        # print(f"Set deep layers, num_layers {num_layers}, bar layer {bar_layer_name}, update layers {self.backward_blocks[:num_layers]} ")
        # print(f"Update parameters {update_param_names}")
        # print(f"Remain parameters {remain_param_names}")
        state_dict = self.state_dict()
        pretrained_state_dict = pretrained.state_dict()
        for name in update_param_names:
            state_dict[name] = pretrained_state_dict[name]
            # print(f"update {name}")

        self.load_state_dict(state_dict)
        return update_param_names, remain_param_names

    def set_shallow_layers(self, num_layers, pretrained):
        assert num_layers <= self.total_blocks

        bar_layer_name = self.forward_blocks[num_layers]
        update_param_names, remain_param_names = [], []
        get_to_bar = False
        for name in self.parameter_names:
            if name.startswith(bar_layer_name):
                get_to_bar = True
            if not get_to_bar:
                update_param_names.append(name)
            else:
                remain_param_names.append(name)
        # print(f"Set shallow layers, num_layers {num_layers}, bar layer {bar_layer_name}, update layers {self.forward_blocks[:num_layers]} ")
        # print(f"Update parameters {update_param_names}")
        # print(f"Remain parameters {remain_param_names}")
        state_dict = self.state_dict()
        pretrained_state_dict = pretrained.state_dict()
        for name in update_param_names:
            state_dict[name] = pretrained_state_dict[name]
            # print(f"update {name}")

        self.load_state_dict(state_dict)
        return update_param_names, remain_param_names

    def config_block_params(self):
        module_params = {}
        for name, module in self.named_modules():
            module_params[name] = 0
            for param in module.parameters():
                module_params[name] += np.prod(param.size())
        # print(module_params)

        block_params = {}
        for bname in self.block_names:
            block_params[bname] = module_params[bname]
        # print(block_params)
        # exit()

        self.forward_block_params = {}
        for idx, name in enumerate(self.forward_blocks):
            self.forward_block_params[name] = 0
            for prior_idx in range(idx):
                self.forward_block_params[name] += block_params[self.forward_blocks[prior_idx]]
        # print("Forward block params: ", self.forward_block_params)

        self.backward_block_params = {}
        for idx, name in enumerate(self.backward_blocks):
            self.backward_block_params[name] = 0
            for prior_idx in range(idx):
                self.backward_block_params[name] += block_params[self.backward_blocks[prior_idx]]
        # print("Backward block params: ", self.backward_block_params)

    def config_block_flops(self):
        self.block_flops = {}
        inshape = self.img_size // 2
        # (2 * ci * k^2 - 1) * h * w * co
        # conv1_flops = (2 * 3 * 3^2 - 1) * inshape * inshape * 64
        conv1_flops = (inshape * inshape) * (3 * 64 * 7**2 + 64 * 2)
        self.block_flops['conv1'] = conv1_flops
        inshape = inshape//2

        # layer1
        for idx in range(self.layer_config[0]):
            name = f"layer1.{idx}"
            self.block_flops[name] = self.layer1[idx].FLOPS((64, inshape, inshape))

        # layer2
        for idx in range(self.layer_config[1]):
            name = f"layer2.{idx}"
            self.block_flops[name] = self.layer2[idx].FLOPS((128, inshape, inshape))
            if idx == 0:
                 inshape /= 2 
       
        # layer3
        for idx in range(self.layer_config[2]):
            name = f"layer3.{idx}"
            self.block_flops[name] = self.layer3[idx].FLOPS((256, inshape, inshape))
            if idx == 0:
                 inshape /= 2
        
        # layer4
        for idx in range(self.layer_config[3]):
            name = f"layer4.{idx}"
            self.block_flops[name] = self.layer4[idx].FLOPS((512, inshape, inshape))
            if idx == 0:
                 inshape /= 2

        fc_flops = 512 * self.num_classes + self.num_classes
        self.block_flops['fc'] = fc_flops

        # print("Block flops: ", self.block_flops)

        self.forward_block_flops = {}
        for idx, name in enumerate(self.forward_blocks):
            self.forward_block_flops[name] = 0
            for prior_idx in range(idx):
                self.forward_block_flops[name] += self.block_flops[self.forward_blocks[prior_idx]]
        # print("Forward block flops: ", self.forward_block_flops)

        self.backward_block_flops = {}
        for idx, name in enumerate(self.backward_blocks):
            self.backward_block_flops[name] = 0
            for prior_idx in range(idx):
                self.backward_block_flops[name] += self.block_flops[self.backward_blocks[prior_idx]]
        # print("Backward block flops: ", self.backward_block_flops)

    def reconfig_block_params(self):
        self.vanilla_forward_block_params = self.forward_block_params
        self.vanilla_backward_block_params = self.backward_block_params

        self.forward_block_params = {}
        for name in self.forward_blocks:
            self.forward_block_params[name] = self.vanilla_forward_block_params[name]

        self.backward_block_params = {}
        for name in self.backward_blocks:
            self.backward_block_params[name] = self.vanilla_backward_block_params[name]

        # print("Reconfig Forward block params: ", self.forward_block_params)
        # print("Reconfig Backward block params: ", self.backward_block_params)

    def reconfig_block_flops(self):
        self.vanilla_forward_block_flops = self.forward_block_flops
        self.vanilla_backward_block_flops = self.backward_block_flops

        self.forward_block_flops = {}
        for name in self.forward_blocks:
            self.forward_block_flops[name] = self.vanilla_forward_block_flops[name]

        self.backward_block_flops = {}
        for name in self.backward_blocks:
            self.backward_block_flops[name] = self.vanilla_backward_block_flops[name]

        # print("Reconfig Forward block flops: ", self.forward_block_params)
        # print("Reconfig Backward block flops: ", self.backward_block_params)



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    # Forward blocks:  ['conv1', 'layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc', 'end']
    # Backward blocks:  ['fc', 'layer4.2', 'layer4.1', 'layer4.0', 'layer3.5', 'layer3.4', 'layer3.3', 'layer3.2', 'layer3.1', 'layer3.0', 'layer2.3', 'layer2.2', 'layer2.1', 'layer2.0', 'layer1.2', 'layer1.1', 'layer1.0', 'conv1', 'start']
    # model.vanilla_forward_blocks = model.forward_blocks
    # model.vanilla_backward_blocks = model.backward_blocks
    # model.forward_blocks = [
    #     'conv1', 'layer1.2',
    #     'layer2.1', 'layer2.3',
    #     'layer3.2', 'layer3.5',
    #     'layer4.1', 'layer4.2',
    #     'fc', 'end'
    # ]
    # model.backward_blocks = [
    #     'fc',
    #     'layer4.2', 'layer4.1',
    #     'layer3.5', 'layer3.2',
    #     'layer2.3', 'layer2.1',
    #     'layer1.2', 'conv1', 'start'
    # ]
    # model.total_blocks = len(model.forward_blocks)

    # model.reconfig_block_params()
    # model.reconfig_block_flops()

    return model

def pruned_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def pruned_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model



def test_resnet18(batch_size, iteration):
    mean_times, mean_throughputs = [], []
    error_times, error_throughputs = [], []
    params, flops = [], []

    for ratio in [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:

        model = pruned_resnet18(num_classes=10, ratio=ratio)

        input_shape = (batch_size, 3, 224, 224)
        input = torch.rand(input_shape) 

        model(input)

        inference_times = []
        throughputs = []
        
        for i in range(iteration):
            start = time.time()
            model(input)
            end = time.time()
            elapse = (end-start) * 1000 / batch_size
            inference_times.append(elapse)
            throughputs.append(1000 / inference_times[-1])

        inf_mean = round(np.mean(inference_times), 4)
        inf_std = round(np.std(inference_times), 4)
        th_mean = round(np.mean(throughputs), 4)
        th_std = round(np.std(throughputs), 4)

        mean_times.append(inf_mean)
        mean_throughputs.append(th_mean)
        error_times.append(inf_std)
        error_throughputs.append(th_std)
        params.append(model.forward_block_params['end'])
        flops.append(model.forward_block_flops['end'])

    print(f"resnet18_prune_params = ", params)
    print(f"resnet18_prune_flops = ", flops)

    print(f"resnet18_prune_time_mean = ", mean_times)
    print(f"resnet18_prune_time_error = ", error_times)
    print(f"resnet18_prune_throughput_mean = ", mean_throughputs)
    print(f"resnet18_prune_throughput_error = ", error_throughputs)

    model = resnet18(num_classes=10, ratio=1)
    model.block_forward_inference_time(batch_size=batch_size, iteration=iteration)
    model.block_backward_inference_time(batch_size=batch_size, iteration=iteration)

def test_resnet34(batch_size, iteration):
    mean_times, mean_throughputs = [], []
    error_times, error_throughputs = [], []
    params, flops = [], []

    for ratio in [0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:

        model = pruned_resnet34(num_classes=10, ratio=ratio)

        input_shape = (batch_size, 3, 224, 224)
        input = torch.rand(input_shape) 

        model(input)

        inference_times = []
        throughputs = []
        
        for i in range(iteration):
            start = time.time()
            model(input)
            end = time.time()
            elapse = (end-start) * 1000 / batch_size
            inference_times.append(elapse)
            throughputs.append(1000 / inference_times[-1])

        inf_mean = round(np.mean(inference_times), 4)
        inf_std = round(np.std(inference_times), 4)
        th_mean = round(np.mean(throughputs), 4)
        th_std = round(np.std(throughputs), 4)

        mean_times.append(inf_mean)
        mean_throughputs.append(th_mean)
        error_times.append(inf_std)
        error_throughputs.append(th_std)
        params.append(model.forward_block_params['end'])
        flops.append(model.forward_block_flops['end'])

    print(f"resnet34_prune_params = ", params)
    print(f"resnet34_prune_flops = ", flops)

    print(f"resnet34_prune_time_mean = ", mean_times)
    print(f"resnet34_prune_time_error = ", error_times)
    print(f"resnet34_prune_throughput_mean = ", mean_throughputs)
    print(f"resnet34_prune_throughput_error = ", error_throughputs)

    model = resnet34(num_classes=10, ratio=1)
    model.block_forward_inference_time(batch_size=batch_size, iteration=iteration)
    model.block_backward_inference_time(batch_size=batch_size, iteration=iteration)


if __name__=="__main__":
    import time
    cuda = False
    torch.set_num_threads(1)
    
    batch_size = 1
    iteration = 5
    
    test_resnet18(batch_size, iteration)