import os, sys
import numpy as np
import matplotlib as mlp
from pdb import set_trace as st
import matplotlib.pyplot as plt
mlp.use('tkagg')
import copy

from alexnet import *
from resnet18 import *
from vgg16 import *

models_name = {
    'resnet18': "ResNet18",
    'resnet34': "ResNet34",
    'resnet50': "ResNet50",
    'vgg16_bn': "VGG16_BN",
    'vgg19_bn': "VGG19_BN",
    'alexnet': "AlexNet",
}

plt.style.use('ggplot')

fig, (axs) = plt.subplots(1,3, figsize=(6,2.2))



model_name = "alexnet"
for model_idx, model_name in enumerate(["alexnet", "resnet18", "vgg16_bn"]):

    prune_params = eval(f"{model_name}_prune_params")
    prune_flops = copy.deepcopy(eval(f"{model_name}_prune_flops"))
    prune_time_mean = eval(f"{model_name}_prune_time_mean")
    prune_throughput_mean = eval(f"{model_name}_prune_throughput_mean")
    prune_time_error = eval(f"{model_name}_prune_time_error")
    prune_throughput_error = eval(f"{model_name}_prune_throughput_error")

    block_forward_params = eval(f"{model_name}_block_forward_params")
    block_forward_flops = copy.deepcopy(eval(f"{model_name}_block_forward_flops"))
    block_forward_time_mean = eval(f"{model_name}_block_forward_time_mean")
    block_forward_throughput_mean = eval(f"{model_name}_block_forward_throughput_mean")
    block_forward_time_error = eval(f"{model_name}_block_forward_time_error")
    block_forward_throughput_error = eval(f"{model_name}_block_forward_throughput_error")

    block_backward_params = eval(f"{model_name}_block_backward_params")
    block_backward_flops = copy.deepcopy(eval(f"{model_name}_block_backward_flops"))
    block_backward_time_mean = eval(f"{model_name}_block_backward_time_mean")
    block_backward_throughput_mean = eval(f"{model_name}_block_backward_throughput_mean")
    block_backward_time_error = eval(f"{model_name}_block_backward_time_error")
    block_backward_throughput_error = eval(f"{model_name}_block_backward_throughput_error")
    
    block_backward_flops = [f/block_backward_flops[-1]*100 for f in block_backward_flops]
    block_forward_flops = [f/block_forward_flops[-1]*100 for f in block_forward_flops]
    prune_flops = [f/prune_flops[-1]*100 for f in prune_flops]

    ax = axs[model_idx]
    ax.errorbar(block_backward_flops, block_backward_time_mean, yerr=block_backward_time_error, label=f"Deep")
    ax.errorbar(block_forward_flops, block_forward_time_mean, yerr=block_forward_time_error, label=f"Shallow")
    ax.errorbar(prune_flops, prune_time_mean, yerr=prune_time_error, label=f"Large-Mag.")
    
    ax.set_title(f"{models_name[model_name]}", size=10)
    
    box = ax.get_position()
    slim = box.height * 0.12
    ax.set_position([box.x0, box.y0 - slim, box.width, box.height + slim])
    
    # locs, labels = ax.xticks()
    labels = ax.get_xticklabels()
    locs = ax.get_xticks()
    locs = [0, 50, 100]

    # text_labels = [f"{l:.0e}" for l in locs]
    # text_labels[1] = "0"
    # ax.set_xticklabels(text_labels)
    ax.set_xticks(locs)
    # text_labels = [l/block_backward_flops[-1]*100 for l in locs]
    text_labels = [f"{l:.0f}" for l in locs]
    ax.set_xticklabels(text_labels)

    # if model_idx == 2:
    #     ax.set_xlabel("FLOPs   ")
    # else:
    #     ax.set_xlabel("FLOPs  ")
    ax.set_xlabel("% FLOPs in TEE")
    if model_idx == 0:
        ax.set_ylabel("Inference\nTime (ms)")

# # https://www.kite.com/python/examples/4997/matplotlib-place-a-legend-outside-of-plot-axes
# for i in range(3):
#     box = axs[i].get_position()
#     slim = box.height * 0.05
#     axs[i].set_position([box.x0, box.y0 - (i+1)*slim, box.width, box.height + slim])

axLine, axLabel = axs[0].get_legend_handles_labels()
plt.figlegend( axLine, axLabel, loc = 'lower center', ncol=6 )

plt.tight_layout()
plt.savefig("time_flops.pdf")