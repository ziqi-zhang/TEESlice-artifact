'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from pdb import set_trace as st
import copy
import numpy as np

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, img_size=32, pretrained=False):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.img_size=img_size

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['vgg16'])
        state_dict = model.state_dict()
        ckp['classifier.weight'] = state_dict['classifier.weight']
        ckp['classifier.bias'] = state_dict['classifier.bias']
        model.load_state_dict(ckp, strict=False)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['vgg16_bn'])
        state_dict = model.state_dict()
        ckp['classifier.weight'] = state_dict['classifier.weight']
        ckp['classifier.bias'] = state_dict['classifier.bias']
        model.load_state_dict(ckp, strict=False)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['vgg19'])
        state_dict = model.state_dict()
        ckp['classifier.weight'] = state_dict['classifier.weight']
        ckp['classifier.bias'] = state_dict['classifier.bias']
        model.load_state_dict(ckp, strict=False)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['vgg19_bn'])
        state_dict = model.state_dict()
        ckp['classifier.weight'] = state_dict['classifier.weight']
        ckp['classifier.bias'] = state_dict['classifier.bias']
        model.load_state_dict(ckp, strict=False)
    return model

if __name__=="__main__":
    import time
    cuda = False
    torch.set_num_threads(1)
    
    model = vgg16_bn(num_classes=10)
    if cuda:
        model = model.cuda()
    batch_size = 32
    iteration = 5
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

    inf_mean = np.mean(inference_times)
    inf_std = np.std(inference_times)

    print(f"BS {batch_size}, ITER {iteration}, mean {inf_mean} ms, std {inf_std} ms")