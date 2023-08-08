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
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'gluon_resnet18', 'ssl_resnet18', 'swsl_resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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

    def conv_layer_FLOPs(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        conv_layer_flops = []
        flops =  spatial_dim * self.in_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        conv_layer_flops.append(flops)
        flops = spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        conv_layer_flops.append(flops)
        if self.downsample is not None:
            flops = spatial_dim * self.in_channels * self.out_channels * 1
            flops += spatial_dim * self.out_channels * 2
            conv_layer_flops.append(flops)
        return conv_layer_flops


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * float(self.stride)**2. * self.in_channels * self.out_channels * 1
        flops += spatial_dim * float(self.stride)**2. * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * self.expansion * 1
        flops += spatial_dim * self.out_channels * self.expansion * 2
        if self.downsample is not None:
            flops += spatial_dim * self.in_channels * self.out_channels * self.expansion * 1
            flops += spatial_dim * self.out_channels * self.expansion * 2
        return flops


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, img_size=64, attack_backbone=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(img_size//8, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.attack_backbone=attack_backbone
        self.layer_config = layers
        self.img_size = img_size
        self.num_classes = num_classes
        self.block_type = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()
        self.config_conv_layer_flops()

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
        print("Forward blocks: ", self.forward_blocks)

        for layer_idx, layers in layers_and_idxs_reverse:
            for block_idx in range(layers, 0, -1):
                name = f"layer{layer_idx}.{block_idx-1}"
                self.backward_blocks.append(name)
        # self.backward_blocks.append('bn1')
        self.backward_blocks.append('conv1')
        self.backward_blocks.append('start')
        print("Backward blocks: ", self.backward_blocks)

        self.parameter_names = []
        for name, _ in self.named_parameters():
            self.parameter_names.append(name)
        self.reverse_parameter_names = copy.deepcopy(self.parameter_names)
        self.reverse_parameter_names.reverse()
        # print("Forward parameters: ", self.parameter_names)
        # print("Backward parameters: ", self.reverse_parameter_names)

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

    def forward(self, x, return_features=False, return_internal=False):
        ends = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # for layer in self.layer1:
        #     x = layer(x)
        x = self.layer1(x)
        if return_internal:
            ends.append(x)

        # for layer in self.layer2:
        #     x = layer(x)
        x = self.layer2(x)
        if return_internal:
            ends.append(x)

        # for layer in self.layer3:
        #     x = layer(x)
        x = self.layer3(x)
        if return_internal:
            ends.append(x)

        # for layer in self.layer4:
        #     x = layer(x)
        x = self.layer4(x)
        if return_internal:
            ends.append(x)

        x = self.avgpool(x)
        # size = x.shape[-1]
        # st()
        # out = F.adaptive_avg_pool2d(x, (size, size))
        x = x.view(x.size(0), -1)
        # x = F.dropout(x, 0.5, self.training, True)
        if self.attack_backbone:
            return x
        x = self.fc(x)

        if return_internal:
            return x, ends
        return x
    
    def load_pretrained(self, fn):
        checkpoint = torch.load(fn)
        state_dict = self.state_dict()
        for k in state_dict.keys():
            state_dict[k] = checkpoint['state_dict'][k]
        self.load_state_dict(state_dict)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        for name, p in self.parameters():
            if "fc" not in name:
                p.requires_grad = False
                print(f"Freeze {name}")
    
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
        print(f"Set deep layers, num_layers {num_layers}, bar layer {bar_layer_name}, update layers {self.backward_blocks[:num_layers]} ")
        print(f"Update parameters {update_param_names}")
        print(f"Remain parameters {remain_param_names}")
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
        print(f"Set shallow layers, num_layers {num_layers}, bar layer {bar_layer_name}, update layers {self.forward_blocks[:num_layers]} ")
        print(f"Update parameters {update_param_names}")
        print(f"Remain parameters {remain_param_names}")
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
        print("Forward block params: ", self.forward_block_params)

        self.backward_block_params = {}
        for idx, name in enumerate(self.backward_blocks):
            self.backward_block_params[name] = 0
            for prior_idx in range(idx):
                self.backward_block_params[name] += block_params[self.backward_blocks[prior_idx]]

        print("Backward block params: ", self.backward_block_params)
    def reconfig_block_params(self):
        self.vanilla_forward_block_params = self.forward_block_params
        self.vanilla_backward_block_params = self.backward_block_params

        self.forward_block_params = {}
        for name in self.forward_blocks:
            self.forward_block_params[name] = self.vanilla_forward_block_params[name]

        self.backward_block_params = {}
        for name in self.backward_blocks:
            self.backward_block_params[name] = self.vanilla_backward_block_params[name]

        print("Reconfig Forward block params: ", self.forward_block_params)
        print("Reconfig Backward block params: ", self.backward_block_params)


    def config_conv_layer_flops(self):
        self.conv_layer_flops = {}
        inshape = self.img_size
        # (2 * ci * k^2 - 1) * h * w * co
        # conv1_flops = (2 * 3 * 3^2 - 1) * inshape * inshape * 64
        conv1_flops = (inshape * inshape) * (3 * 64 * 3**2 + 64 * 2)
        self.conv_layer_flops['conv1'] = conv1_flops
        inshape /= 2

        self.recheck_block_layer_flops = {}
        self.recheck_block_layer_flops['conv1'] = self.conv_layer_flops['conv1']

        # layer1
        for idx in range(self.layer_config[0]):
            name = f"layer1.{idx}"
            layer_convs = self.layer1[idx].conv_layer_FLOPs((64, inshape, inshape))
            if self.block_type == BasicBlock:
                self.conv_layer_flops[f"{name}.conv1"] = layer_convs[0]
                self.conv_layer_flops[f"{name}.conv2"] = layer_convs[1]
                self.recheck_block_layer_flops[name] = self.conv_layer_flops[f"{name}.conv1"] + self.conv_layer_flops[f"{name}.conv2"]
                if len(layer_convs) == 3:
                    self.conv_layer_flops[f"{name}.downsample"] = layer_convs[2]
                    self.recheck_block_layer_flops[name] += self.conv_layer_flops[f"{name}.downsample"]
            else:
                raise NotImplementedError

        # layer2
        for idx in range(self.layer_config[1]):
            name = f"layer2.{idx}"
            layer_convs = self.layer2[idx].conv_layer_FLOPs((128, inshape, inshape))
            if self.block_type == BasicBlock:
                self.conv_layer_flops[f"{name}.conv1"] = layer_convs[0]
                self.conv_layer_flops[f"{name}.conv2"] = layer_convs[1]
                self.recheck_block_layer_flops[name] = self.conv_layer_flops[f"{name}.conv1"] + self.conv_layer_flops[f"{name}.conv2"]
                if len(layer_convs) == 3:
                    self.conv_layer_flops[f"{name}.downsample"] = layer_convs[2]
                    self.recheck_block_layer_flops[name] += self.conv_layer_flops[f"{name}.downsample"]
            else:
                raise NotImplementedError
            if idx == 0:
                 inshape /= 2 

        # layer3
        for idx in range(self.layer_config[2]):
            name = f"layer3.{idx}"
            layer_convs = self.layer3[idx].conv_layer_FLOPs((256, inshape, inshape))
            if self.block_type == BasicBlock:
                self.conv_layer_flops[f"{name}.conv1"] = layer_convs[0]
                self.conv_layer_flops[f"{name}.conv2"] = layer_convs[1]
                self.recheck_block_layer_flops[name] = self.conv_layer_flops[f"{name}.conv1"] + self.conv_layer_flops[f"{name}.conv2"]
                if len(layer_convs) == 3:
                    self.conv_layer_flops[f"{name}.downsample"] = layer_convs[2]
                    self.recheck_block_layer_flops[name] += self.conv_layer_flops[f"{name}.downsample"]
            else:
                raise NotImplementedError
            if idx == 0:
                 inshape /= 2 

        # layer4
        for idx in range(self.layer_config[3]):
            name = f"layer4.{idx}"
            layer_convs = self.layer4[idx].conv_layer_FLOPs((512, inshape, inshape))
            if self.block_type == BasicBlock:
                self.conv_layer_flops[f"{name}.conv1"] = layer_convs[0]
                self.conv_layer_flops[f"{name}.conv2"] = layer_convs[1]
                self.recheck_block_layer_flops[name] = self.conv_layer_flops[f"{name}.conv1"] + self.conv_layer_flops[f"{name}.conv2"]
                if len(layer_convs) == 3:
                    self.conv_layer_flops[f"{name}.downsample"] = layer_convs[2]
                    self.recheck_block_layer_flops[name] += self.conv_layer_flops[f"{name}.downsample"]
            else:
                raise NotImplementedError
            if idx == 0:
                 inshape /= 2 

        fc_flops = 512 * self.num_classes + self.num_classes
        self.conv_layer_flops['fc'] = fc_flops
        self.recheck_block_layer_flops['fc'] = fc_flops

        print("Recheck block layer flops: ", self.recheck_block_layer_flops)
        print("Block flops:               ", self.block_flops)
        print("Conv layer flops: ", self.conv_layer_flops)

    def config_block_flops(self):
        self.block_flops = {}
        inshape = self.img_size
        # (2 * ci * k^2 - 1) * h * w * co
        # conv1_flops = (2 * 3 * 3^2 - 1) * inshape * inshape * 64
        conv1_flops = (inshape * inshape) * (3 * 64 * 3**2 + 64 * 2)
        self.block_flops['conv1'] = conv1_flops
        # stride of conv1 is 2
        inshape /= 2

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

        print("Block flops: ", self.block_flops)

        self.forward_block_flops = {}
        for idx, name in enumerate(self.forward_blocks):
            self.forward_block_flops[name] = 0
            for prior_idx in range(idx):
                self.forward_block_flops[name] += self.block_flops[self.forward_blocks[prior_idx]]
        print("Forward block flops: ", self.forward_block_flops)

        self.backward_block_flops = {}
        for idx, name in enumerate(self.backward_blocks):
            self.backward_block_flops[name] = 0
            for prior_idx in range(idx):
                self.backward_block_flops[name] += self.block_flops[self.backward_blocks[prior_idx]]
        print("Backward block flops: ", self.backward_block_flops)

    def reconfig_block_flops(self):
        self.vanilla_forward_block_flops = self.forward_block_flops
        self.vanilla_backward_block_flops = self.backward_block_flops

        self.forward_block_flops = {}
        for name in self.forward_blocks:
            self.forward_block_flops[name] = self.vanilla_forward_block_flops[name]

        self.backward_block_flops = {}
        for name in self.backward_blocks:
            self.backward_block_flops[name] = self.vanilla_backward_block_flops[name]

        print("Reconfig Forward block flops: ", self.forward_block_flops)
        print("Reconfig Backward block flops: ", self.backward_block_flops)



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet18'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            # st()
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
            # ckp['conv1.weight'] = state_dict['conv1.weight']
        model.load_state_dict(ckp)
    return model

def ssl_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        timm_model = timm.create_model('ssl_resnet18', pretrained=True)
        ckp = timm_model.state_dict()
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
        model.load_state_dict(ckp)
    return model


def gluon_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        timm_model = timm.create_model('gluon_resnet18_v1b', pretrained=True)
        # config = resolve_data_config({}, model=timm_model)
        # transform = create_transform(**config)
        # st()
        ckp = timm_model.state_dict()
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
        model.load_state_dict(ckp)
    return model

def swsl_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        timm_model = timm.create_model('swsl_resnet18', pretrained=True)
        ckp = timm_model.state_dict()
        # ckp = model_zoo.load_url(model_urls['resnet34'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
        model.load_state_dict(ckp, strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    # Forward blocks:  ['conv1', 'layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc', 'end']
    # Backward blocks:  ['fc', 'layer4.2', 'layer4.1', 'layer4.0', 'layer3.5', 'layer3.4', 'layer3.3', 'layer3.2', 'layer3.1', 'layer3.0', 'layer2.3', 'layer2.2', 'layer2.1', 'layer2.0', 'layer1.2', 'layer1.1', 'layer1.0', 'conv1', 'start']
    model.vanilla_forward_blocks = model.forward_blocks
    model.vanilla_backward_blocks = model.backward_blocks
    model.forward_blocks = [
        'conv1', 'layer1.2',
        'layer2.1', 'layer2.3',
        'layer3.2', 'layer3.5',
        'layer4.1', 'layer4.2',
        'fc', 'end'
    ]
    model.backward_blocks = [
        'fc',
        'layer4.2', 'layer4.1',
        'layer3.5', 'layer3.2',
        'layer2.3', 'layer2.1',
        'layer1.2', 'conv1', 'start'
    ]
    model.total_blocks = len(model.forward_blocks)

    model.reconfig_block_params()
    model.reconfig_block_flops()

    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet34'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
            # ckp['conv1.weight'] = state_dict['conv1.weight']
        model.load_state_dict(ckp)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet50'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            # ckp['conv1.weight'] = state_dict['conv1.weight']
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
        model.load_state_dict(ckp)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet101'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        if ckp['conv1.weight'].size(-1) != state_dict['conv1.weight'].size(-1):
            # ckp['conv1.weight'] = state_dict['conv1.weight']
            ckp['conv1.weight'] = ckp['conv1.weight'][:,:,2:-2,2:-2]
        model.load_state_dict(ckp)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet152'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model

def create_teacher(arch, pretrained=True, **kwargs):
    return eval(arch)(pretrained=pretrained, **kwargs)
    
if __name__=="__main__":
    model = resnet18(num_classes=10)
    # print(model)