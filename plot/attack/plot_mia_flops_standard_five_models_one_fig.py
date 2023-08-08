import os, sys
import numpy as np
import copy
import matplotlib as mlp
import matplotlib.pyplot as plt
mlp.use('tkagg')

from results.cifar10_alexnet import *
from results.cifar10_resnet18 import *
from results.cifar10_resnet34 import *
from results.cifar10_vgg16_bn import *
from results.cifar10_vgg19_bn import *
from results.cifar100_vgg16_bn import *
from results.cifar100_vgg19_bn import *
from results.cifar100_resnet18 import *
from results.cifar100_resnet34 import *
from results.cifar100_alexnet import *
from results.stl10_resnet18 import *
from results.stl10_resnet34 import *
from results.stl10_vgg16_bn import *
from results.stl10_vgg19_bn import *
from results.stl10_alexnet import *
from results.utkface_resnet18 import *
from results.utkface_resnet34 import *
from results.utkface_vgg16_bn import *
from results.utkface_vgg19_bn import *
from results.utkface_alexnet import *
from results.standard_models import *
models_name = {
    'resnet18': "ResNet18",
    'resnet34': "ResNet34",
    'vgg16_bn': "VGG16_BN",
    'vgg19_bn': "VGG19_BN",
    'alexnet': "AlexNet"
}


dataset = "CIFAR10"
model = "resnet18"
budget = 50

nettailor_size = 250
shadownet_size = 150

ylabel_fontsize = 11
xtick_fontsize = 11
invisible_xtick_fontsize = 4
ytick_fontsize = 9
legend_fontsize = 14

plt.style.use('ggplot')
fig, (axs) = plt.subplots(5*4,4, figsize=(16, 20),)

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
    for model_idx, model in enumerate(["alexnet", "resnet18", "resnet34", "vgg16_bn", "vgg19_bn"]):
# for dataset in ["CIFAR10"]:
#     for model in ["resnet18"]:
        if dataset == "CIFAR100":
            budget = 500
        else:
            budget = 50
        print(dataset, model)

        block_deep_params = eval(f"{dataset}_{model}_block_deep_params")
        # block_deep_flops = eval(f"{dataset}_{model}_block_deep_flops")
        block_deep_flops = copy.deepcopy(eval(f"standard_{model}_backward_flops"))
        block_deep_gen_gap = x100(eval(f"{dataset}_{model}_block_deep_{budget}_gen_gap"))
        block_deep_conf_gap = x100(eval(f"{dataset}_{model}_block_deep_{budget}_conf_gap"))
        block_deep_top3 = x100(eval(f"{dataset}_{model}_block_deep_{budget}_top3_acc"))
        block_deep_mode0 = x100(eval(f"{dataset}_{model}_block_deep_{budget}_mode0_acc"))
        block_deep_mode3 = x100(eval(f"{dataset}_{model}_block_deep_{budget}_mode3_acc"))
        if model == "vgg16_bn":
            del block_deep_params[-3]
            del block_deep_flops[-3]
            del block_deep_gen_gap[-3]
            del block_deep_conf_gap[-3]
            del block_deep_top3[-3]
            del block_deep_mode0[-3]
            del block_deep_mode3[-3]
        
        block_shallow_params = eval(f"{dataset}_{model}_block_shallow_params")
        # block_shallow_flops = eval(f"{dataset}_{model}_block_shallow_flops")
        block_shallow_flops = eval(f"standard_{model}_forward_flops")
        block_shallow_gen_gap = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_gen_gap"))
        block_shallow_conf_gap = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_conf_gap"))
        block_shallow_top3 = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_top3_acc"))
        block_shallow_mode0 = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_mode0_acc"))
        block_shallow_mode3 = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_mode3_acc"))

        block_large_mag_params = eval(f"{dataset}_{model}_block_large_mag_params")
        # block_large_mag_flops = eval(f"{dataset}_{model}_block_large_mag_flops")
        block_large_mag_flops = eval(f"standard_{model}_prune_flops")
        block_large_mag_gen_gap = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_gen_gap"))
        block_large_mag_conf_gap = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_conf_gap"))
        block_large_mag_top3 = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_top3_acc"))
        block_large_mag_mode0 = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_mode0_acc"))
        block_large_mag_mode3 = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_mode3_acc"))
        
        soter_flops = eval(f"standard_{model}_soter_flops")
        soter_gen_gap = x100(eval(f"{dataset}_{model}_soter_{budget}_gen_gap"))
        soter_conf_gap = x100(eval(f"{dataset}_{model}_soter_{budget}_conf_gap"))
        soter_top3 = x100(eval(f"{dataset}_{model}_soter_{budget}_top3_acc"))
        soter_mode0 = x100(eval(f"{dataset}_{model}_soter_{budget}_mode0_acc"))
        soter_mode3 = x100(eval(f"{dataset}_{model}_soter_{budget}_mode3_acc"))
        
        blackbox_gen_gap = np.mean([block_deep_gen_gap[-1], block_shallow_gen_gap[-1], block_large_mag_gen_gap[-1]])
        blackbox_conf_gap = np.mean([block_deep_conf_gap[-1], block_shallow_conf_gap[-1], block_large_mag_conf_gap[-1]])
        blackbox_mode0 = np.mean([block_deep_mode0[-1], block_shallow_mode0[-1], block_large_mag_mode0[-1]])
        blackbox_mode3 = np.mean([block_deep_mode3[-1], block_shallow_mode3[-1], block_large_mag_mode3[-1]])
        whitebox_gen_gap = np.mean([block_deep_gen_gap[0], block_shallow_gen_gap[0], block_large_mag_gen_gap[0]])
        whitebox_conf_gap = np.mean([block_deep_conf_gap[0], block_shallow_conf_gap[0], block_large_mag_conf_gap[0]])
        whitebox_mode0 = np.mean([block_deep_mode0[0], block_shallow_mode0[0], block_large_mag_mode0[0]])
        whitebox_mode3 = np.mean([block_deep_mode3[0], block_shallow_mode3[0], block_large_mag_mode3[0]])
        
        shadownet_gen_gap = eval(f"{dataset}_{model}_shadownet_{budget}_gen_gap") * 100
        shadownet_conf_gap = eval(f"{dataset}_{model}_shadownet_{budget}_conf_gap") * 100
        shadownet_mode0 = eval(f"{dataset}_{model}_shadownet_{budget}_mode0_acc") * 100
        shadownet_mode3 = eval(f"{dataset}_{model}_shadownet_{budget}_mode3_acc") * 100

        acc = eval(f"{dataset}_{model}_acc")
        # blackbox_ = np.mean([block_deep_acc[-1], block_shallow_acc[-1]])
        # blackbox_fidelity = np.mean([block_deep_fidelity[-1], block_shallow_fidelity[-1]])
        # blackbox_asr = np.mean([block_deep_asr[-1], block_shallow_asr[-1]])
        # whitebox_acc = np.mean([block_deep_acc[0], block_shallow_acc[0]])
        # whitebox_fidelity = np.mean([block_deep_fidelity[0], block_shallow_fidelity[0]])
        # whitebox_asr = np.mean([block_deep_asr[0], block_shallow_asr[0]])
        nettailor_task_param = eval(f"{dataset}_{model}_nettailor_task_param")
        # nettailor_task_flops = eval(f"{dataset}_{model}_nettailor_task_flops")
        nettailor_task_flops = eval(f"standard_{dataset}_{model}_nettailor_task_flops")
        # nettailor_stealing_acc = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_acc")
        # nettailor_stealing_fidelity = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_fidelity")
        # nettailor_stealing_asr = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_asr")
        nettailor_acc = eval(f"{dataset}_{model}_nettailor_acc")
        
        # # unify xaxis, flop across 32x32 and 64x64
        # if model in ["vgg16_bn", "vgg19_bn", "alexnet"] and dataset in ["STL10", "UTKFace"]:
        #     block_deep_flops = eval(f"CIFAR10_{model}_block_deep_flops")
        #     block_shallow_flops = eval(f"CIFAR10_{model}_block_shallow_flops")
        #     block_large_mag_flops = eval(f"CIFAR10_{model}_block_large_mag_flops")
        
        max_flop = block_deep_flops[-1]
        xtick_segs = 4
        x_ticks = [max_flop//xtick_segs*i for i in range(xtick_segs+1)]
        text_labels = convert_flop_ticks(x_ticks)



        ax = axs[model_idx*4+0, dataset_idx]
        ax.plot(block_deep_flops, block_deep_gen_gap, marker='x', label="① Deep", color="blue", zorder=1)
        ax.plot(block_shallow_flops, block_shallow_gen_gap, marker='^', label="② Shallow", color="green", zorder=1)
        ax.plot(block_large_mag_flops, block_large_mag_gen_gap, marker='v', label="③ Large Mag.", color="orange", zorder=1)
        ax.plot(soter_flops, soter_gen_gap, marker='>', label="④ Intermediate", color="fuchsia", zorder=1)
        # ax.scatter(0, shadownet_gen_gap, marker=".", label="⑤ Non-Linear", color="aqua", zorder=5, s=shadownet_size)
        ax.axhline(y=whitebox_gen_gap, label="No-Shield", color="black", linestyle="--", zorder=0)
        ax.axhline(y=blackbox_gen_gap, label="Black-box", color="black", linestyle='dashdot', zorder=0)
        ax.scatter(nettailor_task_flops, 0, marker="*", label="Ours", color="red", zorder=5, s=nettailor_size)
        if dataset_idx == 0:
            ax.set_ylabel("Gen.\nGap", fontsize=ylabel_fontsize)
        ax.tick_params(axis='y', labelsize= ytick_fontsize)
        ax.tick_params(axis='x', labelsize= xtick_fontsize)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([],)

        ax = axs[model_idx*4+1, dataset_idx]
        ax.plot(block_deep_flops, block_deep_conf_gap, marker='x', label="Deep", color="blue", zorder=1)
        ax.plot(block_shallow_flops, block_shallow_conf_gap, marker='^', label="Shallow", color="green", zorder=1)
        ax.plot(block_large_mag_flops, block_large_mag_conf_gap, marker='v', label="Large Mag.", color="orange", zorder=1)
        ax.plot(soter_flops, soter_conf_gap, marker='>', label="Intermediate", color="fuchsia", zorder=1)
        # ax.scatter(0, shadownet_conf_gap, marker=".", label="Non-Linear", color="aqua", zorder=5, s=shadownet_size)
        ax.axhline(y=whitebox_conf_gap, label="No-Shield", color="black", linestyle="--", zorder=0)
        ax.axhline(y=blackbox_conf_gap, label="Black-box", color="black", linestyle='dashdot', zorder=0)
        ax.scatter(nettailor_task_flops, 0, marker="*", label="Ours", color="red", zorder=5, s=nettailor_size)
        if dataset_idx == 0:
            ax.set_ylabel("Conf.\nGap", fontsize=ylabel_fontsize)
        ax.tick_params(axis='y', labelsize= ytick_fontsize)
        ax.tick_params(axis='x', labelsize= xtick_fontsize)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([],)

        ax = axs[model_idx*4+2, dataset_idx]
        ax.plot(block_deep_flops, block_deep_mode0, marker='x', label="Deep", color="blue", zorder=1)
        ax.plot(block_shallow_flops, block_shallow_mode0, marker='^', label="Shallow", color="green", zorder=1)
        ax.plot(block_large_mag_flops, block_large_mag_mode0, marker='v', label="Large Mag.", color="orange", zorder=1)
        ax.plot(soter_flops, soter_mode0, marker='>', label="Intermediate", color="fuchsia", zorder=1)
        # ax.scatter(0, shadownet_mode0, marker=".", label="Non-Linear", color="aqua", zorder=5, s=shadownet_size)
        ax.axhline(y=whitebox_mode0, label="No-Shield", color="black", linestyle="--", zorder=0)
        ax.axhline(y=blackbox_mode0, label="Black-box", color="black", linestyle='dashdot', zorder=0)
        ax.scatter(nettailor_task_flops, 50, marker="*", label="Ours", color="red", zorder=5, s=nettailor_size)
        if dataset_idx == 0:
            ax.set_ylabel("Conf.\nAttack", fontsize=ylabel_fontsize)
        ax.tick_params(axis='y', labelsize= ytick_fontsize)
        ax.tick_params(axis='x', labelsize= xtick_fontsize)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([],)
                
        ax = axs[model_idx*4+3, dataset_idx]
        ax.plot(block_deep_flops, block_deep_mode3, marker='x', label="Deep", color="blue", zorder=1)
        ax.plot(block_shallow_flops, block_shallow_mode3, marker='^', label="Shallow", color="green", zorder=1)
        ax.plot(block_large_mag_flops, block_large_mag_mode3, marker='v', label="Large Mag.", color="orange", zorder=1)
        ax.plot(soter_flops, soter_mode3, marker='>', label="Intermediate", color="fuchsia", zorder=1)
        # ax.scatter(0, shadownet_mode3, marker=".", label="Non-Linear", color="aqua", zorder=5, s=shadownet_size)
        ax.axhline(y=whitebox_mode3, label="No-Shield", color="black", linestyle="--", zorder=0)
        ax.axhline(y=blackbox_mode3, label="Black-box", color="black", linestyle='dashdot', zorder=0)
        ax.scatter(nettailor_task_flops, 50, marker="*", label="Ours", color="red", zorder=5, s=nettailor_size)
        if dataset_idx == 0:
            ax.set_ylabel("Grad.\nAttack", fontsize=ylabel_fontsize)
        ax.tick_params(axis='y', labelsize= ytick_fontsize)
        ax.tick_params(axis='x', labelsize= xtick_fontsize)
        ax.set_xticks(x_ticks)
        # text_labels = [manual_flop_convert(l) for l in x_ticks]
        ax.set_xticklabels([],)
        
        if model_idx == 4:
            ax.set_xticklabels(text_labels,)
            ax.set_xlabel("% FLOPs in TEE", fontsize=ylabel_fontsize)
        
axLine, axLabel = axs[0, 0].get_legend_handles_labels()
axLine[4], axLine[5], axLine[6] = axLine[6], axLine[4], axLine[5]
axLabel[4], axLabel[5], axLabel[6] = axLabel[6], axLabel[4], axLabel[5]
plt.figlegend( axLine, axLabel, loc = 'lower center', ncol=8, labelspacing=0.3, columnspacing=1.0, fontsize=legend_fontsize )

plt.tight_layout()
plt.subplots_adjust(left=0.065, bottom=0.05, right=0.99, top=0.975, wspace=0.13, hspace=0.3)

title_fontsize = 15
plt.text(0.15, 0.98, "CIFAR10", transform=fig.transFigure, fontsize=title_fontsize)
plt.text(0.38, 0.98, "CIFAR100", transform=fig.transFigure, fontsize=title_fontsize)
plt.text(0.625, 0.98, "STL10", transform=fig.transFigure, fontsize=title_fontsize)
plt.text(0.86, 0.98, "UTKFace", transform=fig.transFigure, fontsize=title_fontsize)

y, xmin, xmax = 0.793, 0.025, 1.
line = plt.Line2D((xmin, xmax),(y,y), color="black", linestyle="solid", linewidth=2)
fig.add_artist(line)
y, xmin, xmax = 0.605, 0.025, 1.
line = plt.Line2D((xmin, xmax),(y,y), color="black", linestyle="solid", linewidth=2)
fig.add_artist(line)
y, xmin, xmax = 0.415, 0.025, 1.
line = plt.Line2D((xmin, xmax),(y,y), color="black", linestyle="solid", linewidth=2)
fig.add_artist(line)
y, xmin, xmax = 0.23, 0.025, 1.
line = plt.Line2D((xmin, xmax),(y,y), color="black", linestyle="solid", linewidth=2)
fig.add_artist(line)

# # https://www.kite.com/python/examples/4997/matplotlib-place-a-legend-outside-of-plot-axes
for row_idx, rows in enumerate(axs):
    for col_idx, ax in enumerate(rows):
        box = ax.get_position()
        # bias = box.height * 0.11
        # if row_idx % 4 == 0:
        #     ax.set_position([box.x0, box.y0 - 3*bias, box.width, box.height ])
        # elif row_idx % 4 == 1:
        #     ax.set_position([box.x0, box.y0 - 2*bias, box.width, box.height ])
        # elif row_idx % 4 == 2:
        #     ax.set_position([box.x0, box.y0 - bias, box.width, box.height ])


plt.text(0.005, 0.87, "AlexNet", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)
plt.text(0.005, 0.68, "ResNet18", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)
plt.text(0.005, 0.50, "ResNet34", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)
plt.text(0.005, 0.30, "VGG16_BN", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)
plt.text(0.005, 0.11, "VGG19_BN", transform=fig.transFigure, fontsize=title_fontsize, rotation=90)

# plt.subplot_tool()
# plt.show()
plt.savefig(f"mia_flops_standard_five_models_one_fig.pdf")

