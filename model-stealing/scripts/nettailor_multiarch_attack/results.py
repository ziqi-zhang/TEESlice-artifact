import os, sys
import os.path as osp
import json

root = "models_nettailor/adversary_multiarch_attack/"

dataset_name = "CIFAR10"
victim_model = "vgg19_bn"

queries = [50,100,300,500,1000,3000,5000, 10000, 15000, 20000, 25000, 30000]
print(f"{dataset_name}_{victim_model}_attack_budget = ", queries)

for attack_model in ['resnet18', 'resnet34', 'resnet50', 'alexnet', 'vgg16_bn', 'vgg19_bn']:
    dir = f"victim[{dataset_name}-{victim_model}]-proxy[resnet18]-att[{attack_model}]random"

    accs, fidelity, asr = [], [], []

    for sample in queries:
        path = osp.join(root, dir, f"stealing_eval.{sample}.json")
        with open(path) as f:
            results = json.load(f)
        accs.append(round(results["surrogate_acc"], 2))
        fidelity.append(round(results["surrogate_fidelity"], 2))
        asr.append(round(results["adv_sr"], 2))

    print(f"{dataset_name}_{victim_model}_attack_arch_{attack_model}_accuracy = ", accs)
    print(f"{dataset_name}_{victim_model}_attack_arch_{attack_model}_fidelity = ", fidelity)
    print(f"{dataset_name}_{victim_model}_attack_arch_{attack_model}_asr = ", asr)