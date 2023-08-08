"""
Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from pdb import set_trace as st


__all__ = ['ResNet', 'resnet18', 'resnet34']


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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.avgpool = nn.AvgPool2d(4, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        return nn.ModuleList(layers)

    def forward(self, x):
        ends = []
        x = self.conv1(x)
        # return x
        x = self.bn1(x)
        
        x = self.relu(x)
        # x = self.maxpool(x)

        for layer in self.layer1:
            x = layer(x)
            ends.append(x.detach())

        for layer in self.layer2:
            x = layer(x)
            ends.append(x.detach())

        for layer in self.layer3:
            x = layer(x)
            ends.append(x.detach())

        for layer in self.layer4:
            x = layer(x)
            ends.append(x.detach())
        # return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        # x = F.dropout(x, 0.5, self.training, True)
        x = self.fc(x)

        return x, ends

    def load_pretrained(self, fn):
        checkpoint = torch.load(fn)
        state_dict = self.state_dict()
        for k in state_dict.keys():
            state_dict[k] = checkpoint['state_dict'][k]
        self.load_state_dict(state_dict)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False


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
        model.load_state_dict(ckp)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet34'])
        state_dict = model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        model.load_state_dict(ckp)
    return model

def create_teacher(arch, pretrained=True, **kwargs):
    return eval(arch)(pretrained=pretrained, **kwargs)
    
if __name__=="__main__":
    import time
    cuda = False
    torch.set_num_threads(1)

    model = resnet18(num_classes=10)
    if cuda:
        model = model.cuda()
    batch_size = 64
    iteration = 10
    input_shape = (batch_size, 3, 32, 32)
    input = torch.rand(input_shape)
    if cuda:
        input = input.cuda()

    model(input)

    inference_times = []
    
    for i in range(iteration):
        start = time.time()
        model(input)
        end = time.time()
        elapse = (end-start) * 1000 / batch_size
        inference_times.append(elapse)

    import numpy as np
    inf_mean = np.mean(inference_times)
    inf_std = np.std(inference_times)

    print(f"BS {batch_size}, ITER {iteration}, mean {inf_mean} ms, std {inf_std} ms")