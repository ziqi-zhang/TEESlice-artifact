'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy
from pdb import set_trace as st
import numpy as np

__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, img_size=64):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # new maxpool
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_linear = nn.Linear(256, num_classes)

        self.img_size=img_size
        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()

    def init_layer_config(self):
        self.forward_blocks = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self.forward_blocks.append(name)
        self.forward_blocks.append("last_linear")
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

        # ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40', 'last_linear', 'end']
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
                    output_shape /= module.stride[0]
                block_name = f"features.{name}"
                # print(f"{name} output {output_shape}")
                self.block_flops[block_name] = output_shape**2 * module.in_channels * module.out_channels * module.kernel_size[0]**2
            elif isinstance(module, nn.BatchNorm2d):
                self.block_flops[block_name] += output_shape**2 * module.num_features * 2
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                output_shape /= module.stride
                # print(f"{name} output {output_shape}")
        self.block_flops['last_linear'] = self.last_linear.in_features * self.last_linear.out_features + self.last_linear.out_features
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
        x = self.last_linear(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    if pretrained:
        ckp = model_zoo.load_url(url)
        state_dict = model.state_dict()
        ckp['last_linear.weight'] = state_dict['last_linear.weight']
        ckp['last_linear.bias'] = state_dict['last_linear.bias']
        model.load_state_dict(ckp, strict=False)
    return model

if __name__=="__main__":
    model = alexnet(num_classes=10)