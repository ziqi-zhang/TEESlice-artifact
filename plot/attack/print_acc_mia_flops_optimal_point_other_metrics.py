import os, sys
import numpy as np
import matplotlib as mlp
import copy
from pdb import set_trace as st
import matplotlib.pyplot as plt
mlp.use('tkagg')
import csv
import os.path as osp
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
legend_fontsize = 12

plt.style.use('ggplot')
fig, (axs) = plt.subplots(3*2,4, figsize=(16, 8),)

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
    text_labels = [f"{l:.2f}" for l in text_labels]
    return text_labels
    
deep_attr_configs, shallow_attr_configs, mag_attr_configs, soter_attr_configs = [], [], [], []
nettailor_configs = []

attr = "mode3"
dataset_to_short = {
    "CIFAR10": "C10",
    "CIFAR100": "C100",
    "STL10": "S10",
    "UTKFace": "UTK",
}
model_to_names = {
    "alexnet": "AlexNet",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "vgg16_bn": "VGG16_BN",
    "vgg19_bn": "VGG19_BN"
}

assert attr in ['acc', 'fidelity', 'asr', 'gen_gap', 'conf_gap', 'mode0', 'mode3']
csv_results = []

    
# for model_idx, model in enumerate(["alexnet", "resnet18", "vgg16_bn"]):
for model_idx, model in enumerate(["alexnet", "resnet18", "resnet34", "vgg16_bn", "vgg19_bn"]):

    for dataset_idx, dataset in enumerate(["CIFAR10", "CIFAR100", "STL10", "UTKFace"]):

        if dataset == "CIFAR100":
            budget = 500
        else:
            budget = 50
            
        # print(dataset, model)

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
        soter_params = copy.deepcopy(eval(f"standard_{model}_soter_params"))
        soter_params.reverse()
        soter_flops = [f+p for f, p in zip(soter_flops, soter_params)]


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
        
        
        # ==================================================================================
        # Membership inference
        if dataset == "CIFAR100":
            budget = 500
        else:
            budget = 50
        # print(dataset, model)

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
        
        # soter_flops = eval(f"standard_{model}_soter_flops")
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
        
        # ==================================================================================


        blackbox_acc_threshold = blackbox_acc * (1+0.05)
        blackbox_fidelity_threshold = blackbox_fidelity * (1+0.05)
        blackbox_asr_threshold = blackbox_asr * (1+0.05)
        blackbox_gen_gap_threshold = blackbox_gen_gap * (1+0.05)
        blackbox_conf_gap_threshold = blackbox_conf_gap * (1+0.05)
        blackbox_mode0_threshold = blackbox_mode0 * (1+0.05)
        blackbox_mode3_threshold = blackbox_mode3 * (1+0.05)
        
        def find_optimal_config(metrics, flops, blackbox_thred):
            config = -1
            for m, f in zip(metrics, flops):
                if m < blackbox_thred:
                    config = f / flops[-1] * 100
                    break
            if config == -1:
                config = 100
            return config

        block_deep_acc_optimal_config = find_optimal_config(eval(f"block_deep_{attr}"), block_deep_flops, eval(f"blackbox_{attr}_threshold"))
        block_shallow_acc_optimal_config = find_optimal_config(eval(f"block_shallow_{attr}"), block_shallow_flops, eval(f"blackbox_{attr}_threshold"))
        block_large_mag_acc_optimal_config = find_optimal_config(eval(f"block_large_mag_{attr}"), block_large_mag_flops, eval(f"blackbox_{attr}_threshold"))
        soter_acc_optimal_config = find_optimal_config(eval(f"soter_{attr}"), soter_flops, eval(f"blackbox_{attr}_threshold"))
        nettailor_acc_optimal_config = nettailor_task_flops / block_deep_flops[-1] * 100
        
        deep_attr_configs.append(block_deep_acc_optimal_config)
        shallow_attr_configs.append(block_shallow_acc_optimal_config)
        mag_attr_configs.append(block_large_mag_acc_optimal_config)
        soter_attr_configs.append(soter_acc_optimal_config)
        nettailor_configs.append(nettailor_acc_optimal_config)
        

        print(
            f"{model:<10} {dataset:<10}: "
            f"{block_deep_acc_optimal_config:7.2f}% {block_shallow_acc_optimal_config:7.2f}% {block_large_mag_acc_optimal_config:7.2f}% {soter_acc_optimal_config:7.2f}% {nettailor_acc_optimal_config:7.2f}% "
        )
        csv_results.append(
            [model_to_names[model], dataset_to_short[dataset], f"{block_deep_acc_optimal_config:7.2f}%", f"{block_shallow_acc_optimal_config:7.2f}%", f"{block_large_mag_acc_optimal_config:7.2f}%", f"{soter_acc_optimal_config:7.2f}%", f"{nettailor_acc_optimal_config:7.2f}%"]
        )
        
path = osp.join(f"other_metrics_optimal_config_csv/{attr}.csv")
with open(path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["", "", "Deep", "Shallow", "Large Mag.", "Intermediate", "Ours"]
    )
    writer.writerows(csv_results)