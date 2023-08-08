import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


def register_module_output_shape_resnet(model, input_shape):
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            module.output_shape = input_shape / module.stride[0]
            if module.stride[0] > 1 and "downsample" not in name:
                input_shape = input_shape / module.stride[0]
            print(f"{name} shape {input_shape}")
        elif (isinstance(module, nn.BatchNorm2d)):
            module.output_shape = input_shape

def register_module_output_shape_forward_network(model, input_shape):
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            module.output_shape = input_shape / module.stride[0]
            if module.stride[0] > 1:
                input_shape = input_shape / module.stride[0]
            print(f"{name} shape {input_shape}")
        elif (isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)):
            input_shape = input_shape / module.stride
            module.output_shape = input_shape
            print(f"{name} shape {input_shape}")
        elif (isinstance(module, nn.BatchNorm2d)):
            module.output_shape = input_shape



def weight_prune(
    model,
    pretrained,
    prune_ratio,
    input_shape=32,
):
    total = 0
    for name, module in pretrained.named_modules():
        if ( isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear)):
            total += module.weight.data.numel()


    if "ResNet" in str(type(model)):
        register_module_output_shape_resnet(model, input_shape)
    elif "VGG" in str(type(model)) or "AlexNet" in str(type(model)):
        register_module_output_shape_forward_network(model, input_shape)
    else:
        raise NotImplementedError

    total_flop = 0
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            output_shape = module.output_shape
            conv_flop = output_shape**2 * module.out_channels * module.in_channels * module.kernel_size[0]**2
            bn_flop = output_shape**2 * module.out_channels * 2
            total_flop += conv_flop + bn_flop
            # print(f"{name} shape {input_shape}")
        elif isinstance(module, nn.Linear ):
            flop = module.in_features * module.out_features + module.out_features
            total_flop += flop

    print(f"Total flops: {total_flop}")

    
    conv_weights = torch.zeros(total)
    index = 0
    for name, module in pretrained.named_modules():
        if ( isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear)):
            size = module.weight.data.numel()
            conv_weights[index:(index+size)] = module.weight.data.view(-1).abs().clone()
            index += size
    
    y, i = torch.sort(conv_weights, descending=True)
    # thre_index = int(total * prune_ratio)
    # thre = y[thre_index]
    thre_index = int(total * prune_ratio)
    if prune_ratio == 1:
        thre = y[-1]
    else:
        thre = y[thre_index]
    log = f"Pruning threshold: {thre:.4f}"
    print(log)


    pruned = 0
    pruned_bn = 0
    pruned_flop = 0
    
    zero_flag = False
    
    for (name, module), (p_name, p_module) in zip(model.named_modules(), pretrained.named_modules()):
        if ( isinstance(module, nn.Conv2d)):
            weight_copy = p_module.weight.data.abs().clone()
            mask = weight_copy.lt(thre).float()
            inv_mask = weight_copy.gt(thre).float()

            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            module.weight.data.mul_(mask)
            p_module.weight.data.mul_(inv_mask)
            module.weight.data.add_(p_module.weight.data)

            layer_pruned_flop = (mask.numel() - torch.sum(mask)) * module.output_shape**2 
            layer_flop = mask.numel() * module.output_shape**2 
            layer_remain_flop_ratio = (layer_flop-layer_pruned_flop) / layer_flop
            pruned_flop += layer_pruned_flop

            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})"
            f"\tremain flop {layer_flop-layer_pruned_flop}({layer_flop-layer_pruned_flop}/{layer_flop}={layer_remain_flop_ratio:.2f})")
            print(log)

        elif (isinstance(module, nn.Linear)):
            weight_copy = p_module.weight.data.abs().clone()
            mask = weight_copy.lt(thre).float()
            inv_mask = weight_copy.gt(thre).float()

            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            module.weight.data.mul_(mask)
            p_module.weight.data.mul_(inv_mask)
            module.weight.data.add_(p_module.weight.data)

            layer_pruned_flop = (mask.numel() - torch.sum(mask)) 
            layer_flop = mask.numel() 
            layer_remain_flop_ratio = (layer_flop-layer_pruned_flop) / layer_flop
            pruned_flop += layer_pruned_flop

            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})"
            f"\tremain flop {layer_flop-layer_pruned_flop}({layer_flop-layer_pruned_flop}/{layer_flop}={layer_remain_flop_ratio:.2f})")
            print(log)


        elif (isinstance(module, nn.BatchNorm2d)):
            weight_copy = p_module.weight.data.abs().clone()
            mask = weight_copy.lt(thre).float()
            inv_mask = weight_copy.gt(thre).float()

            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            module.weight.data.mul_(mask)
            module.bias.data.mul_(mask)
            module.running_mean.data.mul_(mask)
            module.running_var.data.mul_(mask)

            p_module.weight.data.mul_(inv_mask)
            p_module.bias.data.mul_(inv_mask)
            p_module.running_mean.data.mul_(inv_mask)
            p_module.running_var.data.mul_(inv_mask)

            module.weight.data.add_(p_module.weight.data)
            module.bias.data.add_(p_module.bias.data)
            module.running_mean.data.add_(p_module.running_mean.data)
            module.running_var.data.add_(p_module.running_var.data)

            layer_pruned_flop = (mask.numel() - torch.sum(mask)) * module.output_shape**2
            layer_flop = mask.numel() * module.output_shape**2
            layer_remain_flop_ratio = (layer_flop-layer_pruned_flop) / layer_flop
            pruned_flop += layer_pruned_flop

            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
            f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})"
            f"\tremain flop {layer_flop-layer_pruned_flop}({layer_flop-layer_pruned_flop}/{layer_flop}={layer_remain_flop_ratio:.2f})"
            )
            print(log)

            
    log = (f"Total conv params: {total}, Pruned conv params: {pruned}, "
    f"Pruned ratio: {pruned/total:.2f}")
    print(log)


    log = (f"Total conv flops: {total_flop}, Pruned conv flops: {pruned_flop}, "
    f"Pruned ratio: {pruned_flop/total_flop:.2f}")
    print(log)

    return model, pruned.item(), total, pruned_flop.item(), total_flop
    

