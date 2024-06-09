import os, sys
import numpy as np
from pdb import set_trace as st
import matplotlib as mlp
import matplotlib.pyplot as plt
mlp.use('tkagg')

import CIFAR100_vgg16_bn
import CIFAR100_resnet34
import CIFAR100_vgg19_bn
import CIFAR100_resnet18

import baseline_CIFAR100_vgg16_bn
import baseline_CIFAR100_resnet34
import baseline_CIFAR100_vgg19_bn
import baseline_CIFAR100_resnet18

models_name = {
    'resnet18': "Backbone (ResNet18)",
    'resnet34': "ResNet34",
    'resnet50': "ResNet50",
    'vgg16_bn': "VGG16_BN",
    'vgg19_bn': "VGG19_BN",
    'alexnet': "AlexNet",
    'hybrid': "Hybrid",
}
models_title_name = {
    'resnet18': "ResNet18",
    'resnet34': "ResNet34",
    'resnet50': "ResNet50",
    'vgg16_bn': "VGG16_BN",
    'vgg19_bn': "VGG19_BN",
    'alexnet': "AlexNet",
    'hybrid': "Hybrid",
}

model_markers = {
    'resnet18': 'D',
    'resnet34': '^',
    'resnet50': 'v',
    'vgg16_bn': '>',
    'vgg19_bn': '<',
    'alexnet': 'x',
    'hybrid': '.',
}
default_linewidth = 1
model_linewidth = {
    'resnet18': 2,
    'hybrid': 2,
    'resnet34': default_linewidth,
    'resnet50': default_linewidth,
    'vgg16_bn': default_linewidth,
    'vgg19_bn': default_linewidth,
    'alexnet': default_linewidth,
}
attr_to_names = {
    "accuracy": "Accuracy",
    "fidelity": "Fidelity",
    "asr": "ASR",
}

dataset = "CIFAR100"
model = "resnet18"
budget = 50
attr = "accuracy"
assert attr in ["accuracy", "fidelity", "asr"]

plt.style.use('ggplot')
fig, (axs) = plt.subplots(2,4, sharex=True, figsize=(15,4))

num_budget = len(CIFAR100_resnet34.CIFAR100_resnet34_attack_budget)
xticks = [0, 5000, 10000, 15000, 20000, 25000, 30000]
xtick_labels = ["0", "5K", "10K", "15K", "20K", "25K", "30K"]

resnet18_ratio = []
resnet34_resnet18_ratio = []

all_baseline_metrics, all_nettailor_metrics = [], []

for dataset_idx, dataset_name in enumerate(["CIFAR100"]):
    for model_idx, model_name in enumerate(["resnet18", "resnet34", "vgg16_bn", "vgg19_bn"]):
        fig_idx = model_idx
        budgets = eval(f"{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_budget")
        
        best_metric = [-1] * num_budget
        best_metric_model, best_metric_cnt = "", -1
        best_baseline_metric = [-1] * num_budget
        best_baseline_metric_model, best_baseline_metric_cnt = "", -1
        
        for attack_model in ["hybrid", "resnet18", "resnet34", "vgg16_bn", "vgg19_bn", "alexnet"]:
        # for attack_model in ["hybrid"]:
            nettailor_name = f"{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_arch_{attack_model}_{attr}"
            baseline_name = f"baseline_{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_arch_{attack_model}_{attr}"
            
            metric = eval(nettailor_name)
            best_metric = [max(metric[i], best_metric[i]) for i in range(num_budget)]
            all_nettailor_metrics += metric
            
            ax = axs[0, fig_idx]
            ax.plot(budgets, metric, marker=model_markers[attack_model], label=models_name[attack_model], linewidth=model_linewidth[attack_model])
            
            baseline_metric = eval(baseline_name)
            best_baseline_metric = [max(baseline_metric[i], best_baseline_metric[i]) for i in range(num_budget)]
            all_baseline_metrics += baseline_metric
            
            ax = axs[1, fig_idx]
            ax.plot(budgets, baseline_metric, marker=model_markers[attack_model], label=models_name[attack_model], linewidth=model_linewidth[attack_model])
            
            if attack_model == "resnet18":
                for nettailor_acc, blackbox_acc in zip(metric, baseline_metric):
                    resnet18_ratio.append(nettailor_acc / blackbox_acc)
            
        axs[0,fig_idx].set_title(f"{dataset_name} {models_title_name[model_name]}")
        
        nettailor_resnet18_name = f"{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_arch_resnet18_{attr}"
        baseline_resnet18_name = f"baseline_{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_arch_resnet18_{attr}"
        resnet18_best_metric, resnet18_best_baseline_metric = 0, 0
        for i in range(num_budget):
            if eval(nettailor_resnet18_name)[i] == best_metric[i]:
                resnet18_best_metric += 1
            if eval(baseline_resnet18_name)[i] == best_baseline_metric[i]:
                resnet18_best_baseline_metric += 1
            
        best_metric_cnt, best_baseline_metric_cnt = 0, 0
        best_metric_model, best_baseline_metric_model = "", ""
        for attack_model in ["hybrid", "resnet18", "resnet34", "vgg16_bn", "vgg19_bn", "alexnet"]:
            nettailor_name = f"{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_arch_{attack_model}_{attr}"
            baseline_name = f"baseline_{dataset_name}_{model_name}.{dataset_name}_{model_name}_attack_arch_{attack_model}_{attr}"
            
            metric = eval(nettailor_name)
            metric_cnt = sum([i==j for i,j in zip(best_metric, metric)])
            if metric_cnt > best_metric_cnt:
                best_metric_cnt = metric_cnt
                best_metric_model = attack_model
                
            metric = eval(baseline_name)
            metric_cnt = sum([i==j for i,j in zip(best_baseline_metric, metric)])
            if metric_cnt > best_baseline_metric_cnt:
                best_baseline_metric_cnt = metric_cnt
                best_baseline_metric_model = attack_model
            

        text_fontsize = 12
        # axs[0,fig_idx].text(0.4, 0.2, f"Top1 is Backbone: {resnet18_best_metric}/{num_budget}", fontsize=text_fontsize, transform=axs[0,fig_idx].transAxes)
        # axs[0,fig_idx].text(0.3, 0.07, f"Top1: {models_title_name[best_metric_model]} ({best_metric_cnt}/{num_budget})", fontsize=text_fontsize, transform=axs[0,fig_idx].transAxes)
        
        # axs[1,fig_idx].text(0.4, 0.2, f"Top1 is Backbone: {resnet18_best_baseline_metric}/{num_budget}", fontsize=text_fontsize, transform=axs[1,fig_idx].transAxes)
        # axs[1,fig_idx].text(0.3, 0.07, f"Top1: {models_title_name[best_baseline_metric_model]} ({best_baseline_metric_cnt}/{num_budget})", fontsize=text_fontsize, transform=axs[1,fig_idx].transAxes)
        
        axs[1,fig_idx].set_xticks(xticks)
        axs[1,fig_idx].set_xticklabels(xtick_labels)
        
        if model_name in ["vgg16_bn", "vgg19_bn", "resnet18", "resnet34"]:
            axs[0,fig_idx].set_yticks([0,25,50,75])
            axs[1,fig_idx].set_yticks([0,25,50,75])

axs[0,0].set_ylabel(f"Our Approach\n{attr_to_names[attr]}")
axs[1,0].set_ylabel(f"Blackbox\n{attr_to_names[attr]}")

axs[1,0].set_xlabel("#Queried Data")
axs[1,1].set_xlabel("#Queried Data")
axs[1,2].set_xlabel("#Queried Data")
axs[1,3].set_xlabel("#Queried Data")

# # https://www.kite.com/python/examples/4997/matplotlib-place-a-legend-outside-of-plot-axes
for i in range(2):
    for j in range(4):
        box = axs[i,j].get_position()
        slim = box.height * 0.08
        axs[i,j].set_position([box.x0, box.y0 - (i+1)*slim, box.width, box.height + slim])


axLine, axLabel = axs[0,0].get_legend_handles_labels()
plt.figlegend( axLine, axLabel, loc = 'lower center', ncol=6, fontsize=12 )
        
plt.tight_layout()
plt.savefig(f"multi_arch_assumption/multi_arch_cifar100_compare_{attr}.pdf")

print(np.mean(resnet18_ratio))

from scipy import stats
res = stats.wilcoxon(all_baseline_metrics, all_nettailor_metrics, alternative="two-sided")
print("two-sided", res)
res = stats.wilcoxon(all_baseline_metrics, all_nettailor_metrics, alternative="greater")
print("greater", res)
res = stats.wilcoxon(all_baseline_metrics, all_nettailor_metrics, alternative="less")
print("less", res)