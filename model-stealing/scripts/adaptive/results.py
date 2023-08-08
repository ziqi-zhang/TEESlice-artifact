import os, sys
import os.path as osp
import json

root = "models_nettailor/adaptive_attack/full_model"

dataset_name = "CIFAR100"
victim_model = "vgg19_bn"

queries = [50,100,300,500,1000,3000,5000, 10000, 15000, 20000, 25000, 30000]


for victim_model in ['resnet18', 'resnet34','vgg16_bn', 'vgg19_bn']:
    dir = f"victim[{dataset_name}-{victim_model}]-proxy[resnet18]-random"

    accs, fidelity, asr = [], [], []

    for sample in queries:
        path = osp.join(root, dir, f"stealing_eval.{sample}.json")
        with open(path) as f:
            results = json.load(f)
        accs.append(round(results["surrogate_acc"], 2))
        fidelity.append(round(results["surrogate_fidelity"], 2))
        asr.append(round(results["adv_sr"], 2))

    print(f"{dataset_name}_{victim_model}_attack_budget = ", queries)
    print(f"{dataset_name}_{victim_model}_adaptive_attack_accuracy = ", accs)
    print(f"{dataset_name}_{victim_model}_adaptive_attack_fidelity = ", fidelity)
    print(f"{dataset_name}_{victim_model}_adaptive_attack_asr = ", asr)