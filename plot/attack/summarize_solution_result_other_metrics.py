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


dataset = "CIFAR10"
model = "resnet18"
budget = 50

nettailor_size = 200
shadownet_size = 100

ylabel_fontsize = 13
xtick_fontsize = 12
invisible_xtick_fontsize = 4
ytick_fontsize = 12
legend_fontsize = 15

plt.style.use('ggplot')
fig, (axs) = plt.subplots(3*3,4, figsize=(16, 12),)

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
    
whitebox_attrs, shadownet_attrs, darknetz_attrs, serdab_attrs, tdsc_attrs, soter_attrs, nettailor_attrs = [], [], [], [], [], [], []
# shadownet_gen_gaps, darknetz_gen_gaps, serdab_gen_gaps, tdsc_gen_gaps, soter_gen_gaps, nettailor_gen_gaps = [], [], [], [], [], []
# whitebox_mode0s, shadownet_mode0s, darknetz_mode0s, serdab_mode0s, tdsc_mode0s, soter_mode0s, nettailor_mode0s = [], [], [], [], [], [], []
csv_results = []

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

for model_idx, model in enumerate(["alexnet", "resnet18", "resnet34", "vgg16_bn", "vgg19_bn"]):
# for model_idx, model in enumerate(["alexnet", "resnet18", "vgg16_bn"]):
# for model_idx, model in enumerate(["resnet34", "vgg19_bn"]):
    for dataset_idx, dataset in enumerate(["CIFAR10", "CIFAR100", "STL10", "UTKFace"]):
        if dataset == "CIFAR100":
            budget = 500
        else:
            budget = 50
            

        block_deep_acc = eval(f"{dataset}_{model}_block_deep_{budget}_acc")
        block_deep_fidelity = eval(f"{dataset}_{model}_block_deep_{budget}_fidelity")
        block_deep_asr = eval(f"{dataset}_{model}_block_deep_{budget}_asr")

        block_shallow_acc = eval(f"{dataset}_{model}_block_shallow_{budget}_acc")
        block_shallow_fidelity = eval(f"{dataset}_{model}_block_shallow_{budget}_fidelity")
        block_shallow_asr = eval(f"{dataset}_{model}_block_shallow_{budget}_asr")

        block_large_mag_acc = eval(f"{dataset}_{model}_block_large_mag_{budget}_acc")
        block_large_mag_fidelity = eval(f"{dataset}_{model}_block_large_mag_{budget}_fidelity")
        block_large_mag_asr = eval(f"{dataset}_{model}_block_large_mag_{budget}_asr")
        
        soter_params = eval(f"{dataset}_{model}_soter_stealing_{budget}_protect_params")
        # soter_flops = eval(f"{dataset}_{model}_soter_stealing_{budget}_protect_flops")
        soter_flops = copy.deepcopy(eval(f"standard_{model}_soter_flops"))
        soter_acc = eval(f"{dataset}_{model}_soter_stealing_{budget}_acc")
        soter_fidelity = eval(f"{dataset}_{model}_soter_stealing_{budget}_fidelity")
        soter_asr = eval(f"{dataset}_{model}_soter_stealing_{budget}_asr")
        
        nettailor_task_flops = eval(f"{dataset}_{model}_nettailor_task_flops")
        nettailor_acc = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_acc")
        nettailor_fidelity = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_fidelity")
        nettailor_asr = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_asr")
        
        
        block_deep_gen_gap = x100(eval(f"{dataset}_{model}_block_deep_{budget}_gen_gap"))
        block_deep_conf_gap = x100(eval(f"{dataset}_{model}_block_deep_{budget}_conf_gap"))
        block_deep_top3 = x100(eval(f"{dataset}_{model}_block_deep_{budget}_top3_acc"))
        block_deep_mode0 = x100(eval(f"{dataset}_{model}_block_deep_{budget}_mode0_acc"))
        block_deep_mode3 = x100(eval(f"{dataset}_{model}_block_deep_{budget}_mode3_acc"))

        block_shallow_gen_gap = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_gen_gap"))
        block_shallow_conf_gap = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_conf_gap"))
        block_shallow_top3 = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_top3_acc"))
        block_shallow_mode0 = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_mode0_acc"))
        block_shallow_mode3 = x100(eval(f"{dataset}_{model}_block_shallow_{budget}_mode3_acc"))

        block_large_mag_gen_gap = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_gen_gap"))
        block_large_mag_conf_gap = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_conf_gap"))
        block_large_mag_top3 = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_top3_acc"))
        block_large_mag_mode0 = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_mode0_acc"))
        block_large_mag_mode3 = x100(eval(f"{dataset}_{model}_block_large_mag_{budget}_mode3_acc"))
        
        shadownet_gen_gap = round(eval(f"{dataset}_{model}_shadownet_{budget}_gen_gap") * 100, 2)
        shadownet_conf_gap = round(eval(f"{dataset}_{model}_shadownet_{budget}_conf_gap") * 100, 2)
        shadownet_top3 = round(eval(f"{dataset}_{model}_shadownet_{budget}_top3_acc") * 100, 2)
        shadownet_mode0 = round(eval(f"{dataset}_{model}_shadownet_{budget}_mode0_acc") * 100, 2)
        shadownet_mode3 = round(eval(f"{dataset}_{model}_shadownet_{budget}_mode3_acc") * 100, 2)
        
        soter_flops = eval(f"standard_{model}_soter_flops")
        soter_gen_gap = x100(eval(f"{dataset}_{model}_soter_{budget}_gen_gap"))
        soter_conf_gap = x100(eval(f"{dataset}_{model}_soter_{budget}_conf_gap"))
        soter_top3 = x100(eval(f"{dataset}_{model}_soter_{budget}_top3_acc"))
        soter_mode0 = x100(eval(f"{dataset}_{model}_soter_{budget}_mode0_acc"))
        soter_mode3 = x100(eval(f"{dataset}_{model}_soter_{budget}_mode3_acc"))
        

        
        acc = eval(f"{dataset}_{model}_acc")
        blackbox_acc = np.mean([block_deep_acc[-1], block_shallow_acc[-1]])
        blackbox_fidelity = np.mean([block_deep_fidelity[-1], block_shallow_fidelity[-1]])
        blackbox_asr = np.mean([block_deep_asr[-1], block_shallow_asr[-1]])
        whitebox_acc = np.mean([block_deep_acc[0], block_shallow_acc[0]])
        whitebox_fidelity = np.mean([block_deep_fidelity[0], block_shallow_fidelity[0]])
        whitebox_asr = np.mean([block_deep_asr[0], block_shallow_asr[0]])
        
        blackbox_gen_gap = np.mean([block_deep_gen_gap[-1], block_shallow_gen_gap[-1]])
        blackbox_conf_gap = np.mean([block_deep_conf_gap[-1], block_shallow_conf_gap[-1]])
        blackbox_mode0 = np.mean([block_deep_mode0[-1], block_shallow_mode0[-1]])
        blackbox_mode3 = np.mean([block_deep_mode3[-1], block_shallow_mode3[-1]])
        whitebox_gen_gap = np.mean([block_deep_gen_gap[0], block_shallow_gen_gap[0]])
        whitebox_conf_gap = np.mean([block_deep_conf_gap[0], block_shallow_conf_gap[0]])
        whitebox_mode0 = np.mean([block_deep_mode0[0], block_shallow_mode0[0]])
        whitebox_mode3 = np.mean([block_deep_mode3[0], block_shallow_mode3[0]])
        
        shadownet_acc = eval(f"{dataset}_{model}_shadownet_stealing_{budget}_acc")
        shadownet_fidelity = eval(f"{dataset}_{model}_shadownet_stealing_{budget}_fidelity")
        shadownet_asr = eval(f"{dataset}_{model}_shadownet_stealing_{budget}_asr")

        darknetz_acc = block_deep_acc[1]
        darknetz_fidelity = block_deep_fidelity[1]
        darknetz_asr = block_deep_asr[1]
        darknetz_gen_gap = block_deep_gen_gap[1]
        darknetz_conf_gap = block_deep_conf_gap[1]
        darknetz_mode0 = block_deep_mode0[1]
        darknetz_mode3 = block_deep_mode3[1]
        
        serdab_acc = block_shallow_acc[1]
        serdab_fidelity = block_shallow_fidelity[1]
        serdab_asr = block_shallow_fidelity[1]
        serdab_gen_gap = block_shallow_gen_gap[1]
        serdab_conf_gap = block_shallow_conf_gap[1]
        serdab_mode0 = block_shallow_mode0[1]
        serdab_mode3 = block_shallow_mode3[1]
        
        
        tdsc_acc = block_large_mag_acc[1]
        tdsc_fidelity = block_large_mag_fidelity[1]
        tdsc_asr = block_large_mag_asr[1]
        tdsc_gen_gap = block_large_mag_gen_gap[1]
        tdsc_conf_gap = block_large_mag_conf_gap[1]
        tdsc_mode0 = block_large_mag_mode0[1]
        tdsc_mode3 = block_large_mag_mode3[1]
        
        
        hyper_soter_acc = soter_acc[1]
        hyper_soter_fidelity = soter_fidelity[1]
        hyper_soter_asr = soter_asr[1]
        hyper_soter_gen_gap = soter_gen_gap[1]
        hyper_soter_conf_gap = soter_conf_gap[1]
        hyper_soter_mode0 = soter_mode0[1]
        hyper_soter_mode3 = soter_mode3[1]
        
        whitebox_attr = round(eval(f"whitebox_{attr}"), 2)
        blackbox_attr = round(eval(f"blackbox_{attr}"), 2)
        darknetz_attr = round(eval(f"darknetz_{attr}"), 2)
        serdab_attr = round(eval(f"serdab_{attr}"), 2)
        tdsc_attr = round(eval(f"tdsc_{attr}"), 2)
        hyper_soter_attr = round(eval(f"hyper_soter_{attr}"), 2)
        shadownet_attr = round(eval(f"shadownet_{attr}"), 2)
        
        if attr in ['acc', 'fidelity', 'asr']:
            nettailor_attr = eval(f"nettailor_{attr}")
        elif attr in ['gen_gap', 'conf_gap']:
            nettailor_attr = blackbox_attr = 0
        else:
            nettailor_attr = blackbox_attr = 50.00
        
        
        print(
            f"{model:<10} {dataset:<10}: "
            f"{whitebox_attr:7.2f} {darknetz_attr:7.2f} {serdab_attr:7.2f} {tdsc_attr:7.2f} {hyper_soter_attr:7.2f} {nettailor_attr:7.2f} {blackbox_attr:7.2f} {'':<5}"
        )
        csv_results.append(
            [model_to_names[model], dataset_to_short[dataset], f"{whitebox_attr}%", f"{darknetz_attr}%", f"{serdab_attr}%", f"{tdsc_attr}%", f"{hyper_soter_attr}%", f"{nettailor_attr}%", f"{blackbox_attr}%"]
        )
        
#         whitebox_attrs.append(whitebox_attr / blackbox_attr)
#         shadownet_attrs.append(shadownet_attr / blackbox_attr)
#         darknetz_attrs.append(darknetz_acc / blackbox_acc)
#         serdab_attrs.append(serdab_attr / blackbox_attr)
#         tdsc_attrs.append(tdsc_attr / blackbox_attr)
#         soter_attrs.append(hyper_soter_attr / blackbox_attr)
#         nettailor_attrs.append(nettailor_attr / blackbox_attr)
        

path = osp.join(f"other_metrics_summarize_solution_csv/{attr}.csv")
with open(path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["", "", "White-box", "DarkneTZ", "Serdab", "Magnitude", "SOTER", "Ours", "Black-box"]
    )
    writer.writerows(csv_results)