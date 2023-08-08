'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
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

    def __init__(self, features, num_classes=1000, img_size=64, pretrained=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.img_size=img_size

        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()
        self.config_conv_layer_flops()


    def init_layer_config(self):
        self.forward_blocks = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self.forward_blocks.append(name)
        self.forward_blocks.append("classifier")
        self.backward_blocks = copy.deepcopy(self.forward_blocks)
        self.backward_blocks.reverse()
        self.total_blocks = len(self.forward_blocks)
        print("Total blocks: ", self.total_blocks)
        self.forward_blocks.append('end')
        self.backward_blocks.append('start')
        print("Forward blocks: ", self.forward_blocks)
        print("Backward blocks: ", self.backward_blocks)

        self.parameter_names = []
        for name, _ in self.named_parameters():
            self.parameter_names.append(name)
        self.reverse_parameter_names = copy.deepcopy(self.parameter_names)
        self.reverse_parameter_names.reverse()
        # print("Forward parameters: ", self.parameter_names)
        # print("Backward parameters: ", self.reverse_parameter_names)
        

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

        # ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40', 'classifier', 'end']
        block_params = {}
        for bname in self.forward_blocks[:-1]:
            block_params[bname] = module_params[bname]
        block_name = None
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                block_name = name
            elif isinstance(module, nn.BatchNorm2d):
                block_params[block_name] += module_params[name]
                # print(f"{name} to {block_name}")

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

    def config_block_flops(self):
        self.block_flops = {}
        output_shape = self.img_size

        block_name = None
        for name, module in self.features.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.stride[0] > 1:
                    output_shape /= 2
                block_name = f"features.{name}"
                # print(f"{name} output {output_shape}")
                self.block_flops[block_name] = output_shape**2 * module.in_channels * module.out_channels * module.kernel_size[0]**2
            elif isinstance(module, nn.BatchNorm2d):
                self.block_flops[block_name] += output_shape**2 * module.num_features * 2
            elif isinstance(module, nn.MaxPool2d):
                output_shape /= module.stride
                # print(f"{name} output {output_shape}")
        self.block_flops['classifier'] = self.classifier.in_features * self.classifier.out_features + self.classifier.out_features
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

    def config_conv_layer_flops(self):
        self.conv_layer_flops = self.block_flops

    def forward(self, x):
        x = self.features(x)
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

    def reconfig_block_flops(self):
        self.vanilla_forward_block_flops = self.forward_block_flops
        self.vanilla_backward_block_flops = self.backward_block_flops

        self.forward_block_flops = {}
        for name in self.forward_blocks:
            self.forward_block_flops[name] = self.vanilla_forward_block_flops[name]

        self.backward_block_flops = {}
        for name in self.backward_blocks:
            self.backward_block_flops[name] = self.vanilla_backward_block_flops[name]

        print("Reconfig Forward block flops: ", self.forward_block_params)
        print("Reconfig Backward block flops: ", self.backward_block_params)



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
    layers.append(nn.AdaptiveAvgPool2d((1,1)))

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

    # Forward blocks:  ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46', 'features.49', 'classifier', 'end']
    # Backward blocks:  ['classifier', 'features.49', 'features.46', 'features.43', 'features.40', 'features.36', 'features.33', 'features.30', 'features.27', 'features.23', 'features.20', 'features.17', 'features.14', 'features.10', 'features.7', 'features.3', 'features.0', 'start']

    model.vanilla_forward_blocks = model.forward_blocks
    model.vanilla_backward_blocks = model.backward_blocks

    model.forward_blocks = [
        'features.0', 'features.10',
        'features.20', 'features.30',
        'features.40', 'features.49', 
        'classifier', 'end'
    ]
    model.backward_blocks = [
        'classifier',
        'features.49', 'features.40',
        'features.30', 'features.20',
        'features.10', 'features.0',
        'start',
    ]
    model.total_blocks = len(model.forward_blocks)

    model.reconfig_block_params()
    model.reconfig_block_flops()

    if pretrained:
        ckp = model_zoo.load_url(model_urls['vgg19_bn'])
        state_dict = model.state_dict()
        ckp['classifier.weight'] = state_dict['classifier.weight']
        ckp['classifier.bias'] = state_dict['classifier.bias']
        model.load_state_dict(ckp, strict=False)
    return model

if __name__=="__main__":
    vgg19_bn()