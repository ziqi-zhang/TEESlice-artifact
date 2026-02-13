import os, sys
import numpy as np
import matplotlib as mlp
import copy
from pdb import set_trace as st
import matplotlib.pyplot as plt
mlp.use('tkagg')

from results.cifar10_resnet18 import *
from results.cifar10_resnet34 import *
from results.cifar10_vgg16_bn import *
from results.cifar10_vgg19_bn import *
from results.cifar100_vgg16_bn import *
from results.cifar100_vgg19_bn import *
from results.cifar100_resnet18 import *
from results.cifar100_resnet34 import *
from results.cifar10_alexnet import *
from results.cifar100_alexnet import *
from results.stl10_resnet18 import *
from results.stl10_vgg16_bn import *
from results.stl10_resnet34 import *
from results.stl10_vgg19_bn import *
from results.stl10_alexnet import *
from results.utkface_resnet18 import *
from results.utkface_vgg16_bn import *
from results.utkface_resnet34 import *
from results.utkface_vgg19_bn import *
from results.utkface_alexnet import *
from results.standard_models import *
models_name = {
    'resnet18': "ResNet18",
    'vgg16_bn': "VGG16_BN",
    'resnet34': "ResNet34",
    'vgg19_bn': "VGG19_BN",
    'alexnet': "AlexNet"
}

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
# mlp.rcParams['axes.unicode_minus'] = False

dataset = "CIFAR10"
model = "resnet18"
budget = 50

nettailor_size = 250
shadownet_size = 150

ylabel_fontsize = 13
xtick_fontsize = 12
invisible_xtick_fontsize = 4
ytick_fontsize = 12
legend_fontsize = 15

plt.style.use('ggplot')
fig, (axs) = plt.subplots(3,4, figsize=(16, 6),)

def manual_flop_convert(v):
    if v == 0:
        return '0'
    if v > 1e7 and v < 1e8:
        return f'{v/1e7:.1f}e7'
    elif v > 1e8 and v < 1e9:
        return f'{v/1e8:.1f}e8'
    elif v > 1e9 and v < 1e10:
        return f'{v/1e9:.1f}e9'
    elif v > 1e10 and v < 1e11:
        return f'{v/1e10:.1f}e10'
    else:
        print(v)
        raise NotImplementedError
    
def x100(raw_a):
    new_a = [round(a*100,2) for a in raw_a]
    return new_a

def convert_flop_ticks(x_ticks):
    text_labels = [l/x_ticks[-1]*100 for l in x_ticks]
    text_labels = [f"{l:.0f}" for l in text_labels]
    return text_labels
    
    
for dataset_idx, dataset in enumerate(["CIFAR10", "CIFAR100", "STL10", "UTKFace"]):
    for model_idx, model in enumerate(["alexnet", "resnet18", "vgg16_bn"]):
        if dataset == "CIFAR100":
            budget = 500
        else:
            budget = 50
            
        print(dataset, model)

        block_deep_params = eval(f"{dataset}_{model}_block_deep_params")
        # block_deep_flops = eval(f"{dataset}_{model}_block_deep_flops")
        block_deep_flops = copy.deepcopy(eval(f"standard_{model}_backward_flops"))
        block_deep_acc = eval(f"{dataset}_{model}_block_deep_{budget}_acc")
        block_deep_fidelity = eval(f"{dataset}_{model}_block_deep_{budget}_fidelity")
        block_deep_asr = eval(f"{dataset}_{model}_block_deep_{budget}_asr")
        # if dataset == "STL10" and model == "vgg16_bn":
        #     st()
        if model in ["vgg16_bn", "vgg19_bn"]:
            del block_deep_params[-3]
            del block_deep_flops[-3]
            del block_deep_acc[-3]
            del block_deep_fidelity[-3]
            del block_deep_asr[-3]
        
        
        block_shallow_params = eval(f"{dataset}_{model}_block_shallow_params")
        # block_shallow_flops = eval(f"{dataset}_{model}_block_shallow_flops")
        block_shallow_flops = eval(f"standard_{model}_forward_flops")
        block_shallow_acc = eval(f"{dataset}_{model}_block_shallow_{budget}_acc")
        block_shallow_fidelity = eval(f"{dataset}_{model}_block_shallow_{budget}_fidelity")
        block_shallow_asr = eval(f"{dataset}_{model}_block_shallow_{budget}_asr")

        block_large_mag_params = eval(f"{dataset}_{model}_block_large_mag_params")
        # block_large_mag_flops = eval(f"{dataset}_{model}_block_large_mag_flops")
        block_large_mag_flops = eval(f"standard_{model}_prune_flops")
        block_large_mag_acc = eval(f"{dataset}_{model}_block_large_mag_{budget}_acc")
        block_large_mag_fidelity = eval(f"{dataset}_{model}_block_large_mag_{budget}_fidelity")
        block_large_mag_asr = eval(f"{dataset}_{model}_block_large_mag_{budget}_asr")
        
        soter_params = eval(f"{dataset}_{model}_soter_stealing_{budget}_protect_params")
        # soter_flops = eval(f"{dataset}_{model}_soter_stealing_{budget}_protect_flops")
        soter_flops = copy.deepcopy(eval(f"standard_{model}_soter_flops"))
        soter_acc = eval(f"{dataset}_{model}_soter_stealing_{budget}_acc")
        soter_fidelity = eval(f"{dataset}_{model}_soter_stealing_{budget}_fidelity")
        soter_asr = eval(f"{dataset}_{model}_soter_stealing_{budget}_asr")


        acc = eval(f"{dataset}_{model}_acc")
        blackbox_acc = np.mean([block_deep_acc[-1], block_shallow_acc[-1]])
        blackbox_fidelity = np.mean([block_deep_fidelity[-1], block_shallow_fidelity[-1]])
        blackbox_asr = np.mean([block_deep_asr[-1], block_shallow_asr[-1]])
        whitebox_acc = np.mean([block_deep_acc[0], block_shallow_acc[0]])
        whitebox_fidelity = np.mean([block_deep_fidelity[0], block_shallow_fidelity[0]])
        whitebox_asr = np.mean([block_deep_asr[0], block_shallow_asr[0]])
        
        shadownet_stealing_acc = eval(f"{dataset}_{model}_shadownet_stealing_{budget}_acc")
        shadownet_stealing_fidelity = eval(f"{dataset}_{model}_shadownet_stealing_{budget}_fidelity")
        shadownet_stealing_asr = eval(f"{dataset}_{model}_shadownet_stealing_{budget}_asr")

        nettailor_task_param = eval(f"{dataset}_{model}_nettailor_task_param")
        # nettailor_task_flops = eval(f"{dataset}_{model}_nettailor_task_flops")
        nettailor_task_flops = eval(f"standard_{dataset}_{model}_nettailor_task_flops")
        nettailor_stealing_acc = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_acc")
        nettailor_stealing_fidelity = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_fidelity")
        nettailor_stealing_asr = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_asr")
        nettailor_acc = eval(f"{dataset}_{model}_nettailor_acc")

        # # unify xaxis, flop across 32x32 and 64x64
        # if model in ["vgg16_bn", "vgg19_bn", "alexnet"] and dataset in ["STL10", "UTKFace"]:
        #     block_deep_flops = eval(f"CIFAR10_{model}_block_deep_flops")
        #     block_shallow_flops = eval(f"CIFAR10_{model}_block_shallow_flops")
        #     block_large_mag_flops = eval(f"CIFAR10_{model}_block_large_mag_flops")
            
        max_flop = block_deep_flops[-1]
        xtick_segs = 4
        x_ticks = [max_flop//xtick_segs*i for i in range(xtick_segs+1)]
        # text_labels = [manual_flop_convert(l) for l in x_ticks]
        # text_labels = [l/x_ticks[-1]*100 for l in x_ticks]
        text_labels = convert_flop_ticks(x_ticks)
        # Convert x_ticks to 100 - %FLOPs
        text_labels = [f"{100 - float(l):.0f}" for l in text_labels]
        
        # Convert all x-axis data: 100 - %FLOPs
        block_deep_flops_plot = [100 - (f/max_flop)*100 for f in block_deep_flops]
        block_shallow_flops_plot = [100 - (f/max_flop)*100 for f in block_shallow_flops]
        block_large_mag_flops_plot = [100 - (f/max_flop)*100 for f in block_large_mag_flops]
        soter_flops_plot = [100 - (f/max_flop)*100 for f in soter_flops]
        nettailor_task_flops_plot = 100 - (nettailor_task_flops/max_flop)*100
        
        # Convert all y-axis data: whitebox_acc - acc
        block_deep_acc_plot = [whitebox_acc - a for a in block_deep_acc]
        block_shallow_acc_plot = [whitebox_acc - a for a in block_shallow_acc]
        block_large_mag_acc_plot = [whitebox_acc - a for a in block_large_mag_acc]
        soter_acc_plot = [whitebox_acc - a for a in soter_acc]
        nettailor_stealing_acc_plot = whitebox_acc - nettailor_stealing_acc
        whitebox_acc_plot = 0
        blackbox_acc_plot = whitebox_acc - blackbox_acc
        
        # Convert x_ticks positions
        x_ticks_plot = [100 - (t/max_flop)*100 for t in x_ticks]

        ax = axs[model_idx, dataset_idx]
        ax.plot(block_deep_flops_plot, block_deep_acc_plot, marker='x', label="① Deep", color="blue", zorder=1)
        ax.plot(block_shallow_flops_plot, block_shallow_acc_plot, marker='^', label="② Shallow", color="green", zorder=1)
        ax.plot(block_large_mag_flops_plot, block_large_mag_acc_plot, marker='v', label="③ Large Mag.", color="orange", zorder=1)
        ax.plot(soter_flops_plot, soter_acc_plot, marker='>', label="④ Intermediate", color="fuchsia", zorder=1)
        ax.scatter(nettailor_task_flops_plot, nettailor_stealing_acc_plot, marker="*", label="Ours", s=nettailor_size, zorder=5, color="red")
        if dataset_idx == 0:
            ax.set_ylabel("Model Security", fontsize=ylabel_fontsize)
        ax.tick_params(axis='y', labelsize= ytick_fontsize)
        ax.tick_params(axis='x', labelsize= xtick_fontsize)
        ax.set_xticks(x_ticks_plot)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Set axis limits with some padding
        ax.set_xlim(-5, 105)
        ax.set_ylim(-0.05, max([max(block_deep_acc_plot), max(block_shallow_acc_plot), max(block_large_mag_acc_plot), max(soter_acc_plot), nettailor_stealing_acc_plot]) + 8)
        if model_idx == 2:
            ax.set_xlabel("Computation Speed", fontsize=ylabel_fontsize)
        else:
            pass


plt.tight_layout()
plt.subplots_adjust(left=0.065, bottom=0.13, right=0.99, top=0.96, wspace=0.13, hspace=0.3)

title_fontsize = 15
plt.text(0.15, 0.97, "CIFAR10", transform=fig.transFigure, fontsize=title_fontsize)
plt.text(0.38, 0.97, "CIFAR100", transform=fig.transFigure, fontsize=title_fontsize)
plt.text(0.625, 0.97, "STL10", transform=fig.transFigure, fontsize=title_fontsize)
plt.text(0.86, 0.97, "UTKFace", transform=fig.transFigure, fontsize=title_fontsize)

y, xmin, xmax = 0.685, 0.025, 1.
line = plt.Line2D((xmin, xmax),(y,y), color="black", linestyle="solid", linewidth=1.5)
fig.add_artist(line)
y, xmin, xmax = 0.395, 0.025, 1.
line = plt.Line2D((xmin, xmax),(y,y), color="black", linestyle="solid", linewidth=1.5)
fig.add_artist(line)

# https://www.kite.com/python/examples/4997/matplotlib-place-a-legend-outside-of-plot-axes
for row_idx, rows in enumerate(axs):
    for col_idx, ax in enumerate(rows):
        box = ax.get_position()
        # bias = box.height * 0.05
        # ax.set_position([box.x0, box.y0 - row_idx*bias, box.width, box.height ])
        # if row_idx % 2 == 0:
        #     ax.set_position([box.x0, box.y0 - 2*bias, box.width, box.height ])
        # elif row_idx % 2 == 1:
        #     ax.set_position([box.x0, box.y0 - bias, box.width, box.height ])

plt.text(0.005, 0.795, "AlexNet", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)
plt.text(0.005, 0.48, "ResNet18", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)
plt.text(0.005, 0.17, "VGG16_BN", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)

# plt.subplot_tool()
# plt.show()
plt.savefig(f"acc_flops_jobtalk.pdf")
