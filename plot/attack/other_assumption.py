import os, sys
import numpy as np
import matplotlib as mlp
import copy
from pdb import set_trace as st
import matplotlib.pyplot as plt
mlp.use('tkagg')
import os.path as osp
import csv

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

attr = "fidelity"
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
    
whitebox_accs, shadownet_accs, darknetz_accs, serdab_accs, tdsc_accs, soter_accs, nettailor_accs = [], [], [], [], [], [], []
shadownet_gen_gaps, darknetz_gen_gaps, serdab_gen_gaps, tdsc_gen_gaps, soter_gen_gaps, nettailor_gen_gaps = [], [], [], [], [], []
whitebox_mode0s, shadownet_mode0s, darknetz_mode0s, serdab_mode0s, tdsc_mode0s, soter_mode0s, nettailor_mode0s = [], [], [], [], [], [], []

csv_results = []

for model_idx, model in enumerate(["alexnet", "resnet18", "resnet34", "vgg16_bn", "vgg19_bn"]):
# for model_idx, model in enumerate(["alexnet", "resnet18", "vgg16_bn"]):
# for model_idx, model in enumerate(["resnet34", "vgg19_bn"]):
    for dataset_idx, dataset in enumerate(["CIFAR10", "CIFAR100", "STL10", "UTKFace"]):
        if dataset == "CIFAR100":
            budget = 500
        else:
            budget = 50
            


        hybrid_acc = eval(f"{dataset}_{model}_nettailor_stealing_hybrid_{budget}_{attr}")
        hybrid_fidelity = eval(f"{dataset}_{model}_nettailor_stealing_hybrid_{budget}_fidelity")
        hybrid_asr = eval(f"{dataset}_{model}_nettailor_stealing_hybrid_{budget}_asr")
        
        victim_acc = eval(f"{dataset}_{model}_nettailor_stealing_victim_{budget}_{attr}")
        victim_fidelity = eval(f"{dataset}_{model}_nettailor_stealing_victim_{budget}_fidelity")
        victim_asr = eval(f"{dataset}_{model}_nettailor_stealing_victim_{budget}_asr")
        
        backbone_acc = eval(f"{dataset}_{model}_nettailor_stealing_backbone_{budget}_{attr}")
        backbone_fidelity = eval(f"{dataset}_{model}_nettailor_stealing_backbone_{budget}_fidelity")
        backbone_asr = eval(f"{dataset}_{model}_nettailor_stealing_backbone_{budget}_asr")

        print(
            f"{model:<10} {dataset:<10}: "
            f"{hybrid_acc:.2f} {backbone_acc:.2f}{'':<5} {victim_acc:.2f} "
        )
        
        csv_results.append(
            [model_to_names[model], dataset_to_short[dataset], f"{hybrid_acc}%", f"{backbone_acc}%", f"{victim_acc}%",]
        )

path = osp.join(f"other_assumption_csv/{attr}.csv")
with open(path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["", "", "Hybrid","Backbone", "Victim"]
    )
    writer.writerows(csv_results)