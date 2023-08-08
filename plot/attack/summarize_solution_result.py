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
    
whitebox_accs, shadownet_accs, darknetz_accs, serdab_accs, tdsc_accs, soter_accs, nettailor_accs = [], [], [], [], [], [], []
shadownet_gen_gaps, darknetz_gen_gaps, serdab_gen_gaps, tdsc_gen_gaps, soter_gen_gaps, nettailor_gen_gaps = [], [], [], [], [], []
whitebox_mode0s, shadownet_mode0s, darknetz_mode0s, serdab_mode0s, tdsc_mode0s, soter_mode0s, nettailor_mode0s = [], [], [], [], [], [], []


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
        nettailor_stealing_acc = eval(f"{dataset}_{model}_nettailor_stealing_{budget}_acc")
        
        
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
        

        
        blackbox_gen_gap = np.mean([block_deep_gen_gap[-1], block_shallow_gen_gap[-1]])
        blackbox_mode0 = np.mean([block_deep_mode0[-1], block_shallow_mode0[-1]])
        whitebox_gen_gap = np.mean([block_deep_gen_gap[0], block_shallow_gen_gap[0]])
        whitebox_mode0 = np.mean([block_deep_mode0[0], block_shallow_mode0[0]])


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

        darknetz_acc = block_deep_acc[1]
        darknetz_fidelity = block_deep_fidelity[1]
        darknetz_asr = block_deep_asr[1]
        
        serdab_acc = block_shallow_acc[1]
        serdab_fidelity = block_shallow_fidelity[1]
        serdab_asr = block_shallow_fidelity[1]
        
        tdsc_acc = block_large_mag_acc[1]
        tdsc_fidelity = block_large_mag_fidelity[1]
        tdsc_asr = block_large_mag_asr[1]
        
        
        
        hyper_soter_acc = soter_acc[1]
        hyper_soter_fidelity = soter_fidelity[1]
        hyper_soter_asr = soter_asr[1]
        
        darknetz_gen_gap = block_deep_gen_gap[1]
        darknetz_mode0 = block_deep_mode0[1]
        
        serdab_gen_gap = block_shallow_gen_gap[1]
        serdab_mode0 = block_shallow_mode0[1]
        
        tdsc_gen_gap = block_large_mag_gen_gap[1]
        tdsc_mode0 = block_large_mag_mode0[1]
        
        hyper_soter_gen_gap = soter_gen_gap[1]
        hyper_soter_mode0 = soter_mode0[1]
        
        
        # print(
        #     f"{model:<10} {dataset:<10}: "
        #     f"{whitebox_acc:.2f} {shadownet_stealing_acc} {darknetz_acc:.2f} {tdsc_acc:.2f} {'':<5}"
        #     f"{whitebox_gen_gap:.2f} {shadownet_gen_gap} {darknetz_gen_gap:.2f} {tdsc_gen_gap:.2f} {'':<5}"
        #     f"{whitebox_mode0:.2f} {shadownet_mode0} {darknetz_mode0:.2f} {tdsc_mode0:.2f} {'':<5}"
        # )
        print(
            f"{model:<10} {dataset:<10}: "
            f"{whitebox_acc:.2f} {darknetz_acc:.2f} {serdab_acc:.2f} {tdsc_acc:.2f} {hyper_soter_acc:.2f} {shadownet_stealing_acc:.2f} {nettailor_stealing_acc:.2f} {blackbox_acc:.2f} {'':<5}"
            f"{whitebox_mode0:.2f} {darknetz_mode0:.2f} {serdab_mode0:.2f} {tdsc_mode0:.2f} {hyper_soter_mode0:.2f} {shadownet_mode0:.2f} {50.00} {blackbox_mode0:.2f} {'':<5}"
        )
        
        whitebox_accs.append(whitebox_acc / blackbox_acc)
        shadownet_accs.append(shadownet_stealing_acc / blackbox_acc)
        darknetz_accs.append(darknetz_acc / blackbox_acc)
        serdab_accs.append(serdab_acc / blackbox_acc)
        tdsc_accs.append(tdsc_acc / blackbox_acc)
        soter_accs.append(hyper_soter_acc / blackbox_acc)
        nettailor_accs.append(nettailor_stealing_acc / blackbox_acc)
        
        shadownet_gen_gaps.append(shadownet_gen_gap / blackbox_gen_gap)
        darknetz_gen_gaps.append(darknetz_gen_gap / blackbox_gen_gap)
        tdsc_gen_gaps.append(tdsc_gen_gap / blackbox_gen_gap)
        
        # shadownet_mode0s.append((shadownet_mode0-50) / (whitebox_mode0-50))
        # darknetz_mode0s.append((darknetz_mode0-50) / (whitebox_mode0-50))
        # tdsc_mode0s.append((tdsc_mode0-50) / (whitebox_mode0-50))
        
        whitebox_mode0s.append(whitebox_mode0 / blackbox_mode0)
        shadownet_mode0s.append((shadownet_mode0) / (blackbox_mode0))
        darknetz_mode0s.append((darknetz_mode0) / (blackbox_mode0))
        serdab_mode0s.append(serdab_mode0 / blackbox_mode0)
        tdsc_mode0s.append((tdsc_mode0) / (blackbox_mode0))
        soter_mode0s.append(hyper_soter_mode0 / blackbox_mode0)
        nettailor_mode0s.append(50 / blackbox_mode0)
        
# print(
#     f"{'':<10} {'':<10}: "
#     f"{0.:.2f} {np.mean(shadownet_accs):.2f} {np.mean(darknetz_accs):.2f} {np.mean(tdsc_accs):.2f} {'':<5}"
#     f"{0.:.2f} {np.mean(shadownet_gen_gaps):.2f} {np.mean(darknetz_gen_gaps):.2f} {np.mean(tdsc_gen_gaps):.2f} {'':<5}"
#     f"{0.:.2f} {np.mean(shadownet_mode0s):.2f} {np.mean(darknetz_mode0s):.2f} {np.mean(tdsc_mode0s):.2f} {'':<5}"
# )

print(
    f"{'':<10} {'':<10}: "
    f"{np.mean(whitebox_accs):.2f} {np.mean(darknetz_accs):.2f} {np.mean(serdab_accs):.2f} {np.mean(tdsc_accs):.2f} {np.mean(soter_accs):.2f} {np.mean(nettailor_accs):.2f} {'':<5}"
    f"{np.mean(whitebox_mode0s):.2f} {np.mean(darknetz_mode0s):.2f} {np.mean(serdab_mode0s):.2f} {np.mean(tdsc_mode0s):.2f} {np.mean(soter_mode0s):.2f} {np.mean(nettailor_mode0s):.2f} {'':<5}"
)
print(len(soter_accs))