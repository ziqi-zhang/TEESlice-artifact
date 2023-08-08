import os
import json
import pandas as pd

# resnet18: 21 layers
# vgg16_bn: 14 layers
# alexnet: 6 layers

res_dir = "models/soter-var"
dataset_list = ['CIFAR10', 'CIFAR100', 'STL10', 'UTKFaceRace']
model_list = ['resnet18', 'vgg16_bn', 'alexnet', 'resnet34', 'vgg19_bn']
theta_dict = {
    'resnet18': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
    'vgg16_bn': [1.0, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
    'alexnet': [1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.0], 
    'resnet34': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
    'vgg19_bn': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
}

budget_dict = {'CIFAR10': 50, 'CIFAR100': 500, 'STL10': 50, 'UTKFaceRace': 50}
lr = 0.001

res_dict = {}
for ds in dataset_list:
    for mdl in model_list:

        for theta in theta_dict[mdl]:
            cur_dir = os.path.join(res_dir, f'{ds}-{mdl}-theta{theta}-recover')
            res_fname = f'stealing_eval.0.{budget_dict[ds]}.{lr}.json'
            res_path = os.path.join(cur_dir, res_fname)
            with open(res_path) as json_file:
                json_dict = json.load(json_file)

            # victim accuracy
            res_key = f'{ds}_{mdl}_acc'
            if res_key not in res_dict.keys():
                res_dict[res_key] = []
            res_dict[res_key].append(json_dict['trained_acc'])

            # protected params
            res_key = f'{ds}_{mdl}_soter_stealing_{budget_dict[ds]}_protect_params'
            if res_key not in res_dict.keys():
                res_dict[res_key] = []
            res_dict[res_key].append(json_dict['protect_params'])

            # protected flops
            res_key = f'{ds}_{mdl}_soter_stealing_{budget_dict[ds]}_protect_flops'
            if res_key not in res_dict.keys():
                res_dict[res_key] = []
            res_dict[res_key].append(json_dict['protect_flops'])

            # accuracy
            res_key = f'{ds}_{mdl}_soter_stealing_{budget_dict[ds]}_acc'
            if res_key not in res_dict.keys():
                res_dict[res_key] = []
            res_dict[res_key].append(json_dict['surrogate_acc'])

            

            # fidelity
            res_key = f'{ds}_{mdl}_soter_stealing_{budget_dict[ds]}_fidelity'
            if res_key not in res_dict.keys():
                res_dict[res_key] = []
            res_dict[res_key].append(json_dict['surrogate_fidelity'])

            # asr
            res_key = f'{ds}_{mdl}_soter_stealing_{budget_dict[ds]}_asr'
            if res_key not in res_dict.keys():
                res_dict[res_key] = []
            res_dict[res_key].append(json_dict['adv_sr'])

# theta_cols = []
# for theta in theta_list:
#     theta_cols.append(f'theta {theta}')
# res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=theta_cols)
# print(res_df)

# data_dir = 'images'
# res_df.to_csv(os.path.join(data_dir, 'soter_res.csv'))

for k, arr in res_dict.items():
    if ('acc' in k) or ('fidelity' in k) or ('asr' in k):
        print(f'{k} = [', ', '.join('{:0.2f}'.format(i) for i in arr), "]")
    else:
        print(f'{k} = ', arr)
