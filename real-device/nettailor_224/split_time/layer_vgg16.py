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

# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (5): ReLU(inplace=True)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (9): ReLU(inplace=True)
#     (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (12): ReLU(inplace=True)
#     (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (16): ReLU(inplace=True)
#     (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (19): ReLU(inplace=True)
#     (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (26): ReLU(inplace=True)
#     (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (29): ReLU(inplace=True)
#     (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (32): ReLU(inplace=True)
#     (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (36): ReLU(inplace=True)
#     (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (39): ReLU(inplace=True)
#     (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (42): ReLU(inplace=True)
#     (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (classifier): Linear(in_features=512, out_features=10, bias=True)
# )

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, img_size=224, pretrained=False):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.img_size=img_size
        self.num_classes = num_classes


        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()
        self.collect_to_sequential()


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

    def collect_to_sequential(self):
        self.block_input_shapes = [
            (3, 224, 224),
        ]
        channel, shape = 3, 224
        self.block_layers = []
        block = []
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                block.append(module)
                block = nn.Sequential(*block)
                self.block_layers.append(block)
                block = []
                assert module.stride[0] == 1
                channel = module.out_channels

                self.block_input_shapes.append(
                    (channel, shape, shape)
                )
            else:
                block.append(module)

            if isinstance(module, nn.MaxPool2d):
                shape = shape // module.stride

        block += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.num_classes)
        ]
        block = nn.Sequential(*block)
        self.block_layers.append(block)

        self.num_blocks = len(self.block_layers)
        self.block_layers = nn.Sequential(
            *self.block_layers
        )

        # print(self.block_layers)
        # print(self.block_input_shapes)
        
    def block_forward_inference_time(self, batch_size=1, iteration=5):
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

            print(f"Forward block {block_idx+1} Time mean {inf_mean:.4f} ms, std {inf_std:.4f} ms. Throughput mean {th_mean:.4f}, std {th_std:.4f}")

        forward_params = list(self.forward_block_params.values())[1:]
        forward_flops = list(self.forward_block_flops.values())[1:]

        print(f"vgg16_bn_block_forward_params = ", forward_params)
        print(f"vgg16_bn_block_forward_flops = ", forward_flops)

        print(f"vgg16_bn_block_forward_time_mean = ", mean_times)
        print(f"vgg16_bn_block_forward_time_error = ", error_times)
        print(f"vgg16_bn_block_forward_throughput_mean = ", mean_throughputs)
        print(f"vgg16_bn_block_forward_throughput_error = ", error_throughputs)

    def block_backward_inference_time(self, batch_size=1, iteration=5):
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

            print(f"Backward block {block_idx+1} Time mean {inf_mean:.4f} ms, std {inf_std:.4f} ms. Throughput mean {th_mean:.4f}, std {th_std:.4f}")

        backward_params = list(self.backward_block_params.values())[1:]
        backward_flops = list(self.backward_block_flops.values())[1:]

        print(f"vgg16_bn_block_backward_params = ", backward_params)
        print(f"vgg16_bn_block_backward_flops = ", backward_flops)

        print(f"vgg16_bn_block_backward_time_mean = ", mean_times)
        print(f"vgg16_bn_block_backward_time_error = ", error_times)
        print(f"vgg16_bn_block_backward_throughput_mean = ", mean_throughputs)
        print(f"vgg16_bn_block_backward_throughput_error = ", error_throughputs)

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



def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

if __name__=="__main__":
    import time
    cuda = False
    torch.set_num_threads(1)
    
    model = vgg16_bn(num_classes=10)
    # print(model)
    
    with torch.no_grad():
        model.block_backward_inference_time()