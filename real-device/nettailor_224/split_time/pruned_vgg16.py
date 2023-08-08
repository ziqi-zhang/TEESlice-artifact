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


class Pruned_VGG16(nn.Module):

    def __init__(self, features=None, num_classes=1000, img_size=224):
        super(Pruned_VGG16, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(512*ratio), num_classes)
        self._initialize_weights()
        self.img_size=img_size


        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()


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


def pruned_make_layers(cfg, ratio, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, int(v*ratio), kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v*ratio)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v*ratio)
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def pruned_vgg16_bn(pretrained=False, ratio=0.5, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = Pruned_VGG16(pruned_make_layers(cfg['D'], ratio, batch_norm=True), **kwargs)

    return model


if __name__=="__main__":
    import time
    cuda = False
    torch.set_num_threads(1)

    batch_size = 8
    iteration = 5
    
    mean_times, mean_throughputs = [], []
    error_times, error_throughputs = [], []
    params, flops = [], []

    for ratio in [0.2, 0.5, 1.0]:

        model = pruned_vgg16_bn(num_classes=10, ratio=ratio)

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

        inf_mean = np.mean(inference_times)
        inf_std = np.std(inference_times)
        th_mean = np.mean(throughputs)
        th_std = np.std(throughputs)

        mean_times.append(inf_mean)
        mean_throughputs.append(th_mean)
        error_times.append(inf_std)
        error_throughputs.append(th_std)
        params.append(model.forward_block_params['end'])
        flops.append(model.forward_block_flops['end'])

    print(f"vgg16_bn_prune_params = ", params)
    print(f"vgg16_bn_prune_flops = ", flops)

    print(f"vgg16_bn_prune_time_mean = ", mean_times)
    print(f"vgg16_bn_prune_time_error = ", error_times)
    print(f"vgg16_bn_prune_throughput_mean = ", mean_throughputs)
    print(f"vgg16_bn_prune_throughput_error = ", error_throughputs)