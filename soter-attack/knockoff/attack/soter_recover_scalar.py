#!/usr/bin/python

import argparse
from email.contentmanager import raw_data_manager
import os.path as osp
import os
from datetime import datetime
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import knockoff.config as cfg
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import knockoff.models.zoo as zoo

import random
from matplotlib import pyplot as plt
import math
import pandas as pd
from ipdb import set_trace
import copy
import timm
import sys
from functools import partial

import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet

from knockoff.attack.utils_eval import eval_surrogate_model
from knockoff.attack.utils_model import get_stored_net
from collections import OrderedDict


def load_models(params, num_classes):
    """
    Load public known model and trained victim model
    Here we assume that the two models have the same num_classes and model_arch
    [NOTE] CIFAR100 trained from scratch is not considered
    [NOTE] assume the ori_model is pretrained or stored in `downloaded_models` or timm
    """
    model_arch = params['model_arch']
    model_name = params['model_name']
    # Load public known model
    ori_dataset_name = params['ori_dataset']
    ori_modelfamily = datasets.dataset_to_modelfamily[ori_dataset_name]
    pretrained = params['pretrained']
    ori_model = get_stored_net(model_name=model_name, model_arch=model_arch, \
        model_family=ori_modelfamily, pretrained=pretrained, num_classes=num_classes, \
        pretrained_path=params['ori_ckpt_path'])
    
    # Load trained victim model
    trained_dataset_name = params['trained_dataset']
    trained_modelfamily = datasets.dataset_to_modelfamily[trained_dataset_name]
    trained_model, trained_acc = get_stored_net(model_name=model_name, model_arch=model_arch, \
        model_family=trained_modelfamily, pretrained=None, num_classes=num_classes, \
        ckpt_path=params['trained_ckpt_path'], ret_acc=True)
    
    return ori_model, trained_model, trained_acc



class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None, use_STL10=False):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

        self.use_STL10 = use_STL10

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # [NOTE] In stl10.py (L126-128): transpose the image
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.use_STL10:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None, use_STL10=False):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform, \
            use_STL10=use_STL10)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read original and trained models')
    # Common parameters
    parser.add_argument('model_name', metavar='MODEL_NAME', type=str, help='Model name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model architecture')
    parser.add_argument('layer_theta', type=float, default=0.8, metavar='THETA',
                        help='ratio of layers scaled with scalar')
    parser.add_argument('out_dir', metavar='PATH', type=str, help='Results output directory', default=None)
    # Original model
    parser.add_argument('--ori_dataset', metavar='DS_NAME', type=str, help='Original dataset name', default=None)
    parser.add_argument('--ori_ckpt_path', metavar='PATH', type=str, help='Original model checkpoint path', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    # Trained model
    parser.add_argument('--trained_dataset', metavar='DS_NAME', type=str, help='Trained dataset name', default=None)
    parser.add_argument('--trained_ckpt_path', metavar='PATH', type=str, help='Trained model checkpoint path', default=None)
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('--img_dir', type=str, help='Output directory for images', default='images')
    parser.add_argument('--scalar_start', type=float, help='Start of scalar random range', default=1.0)
    parser.add_argument('--scalar_end', type=float, help='End of scalar random range', default=5.0)
    parser.add_argument('--max_cnt', type=int, help='Number of maximum elements to estimate the scalar', default=100)
    parser.add_argument('--repeat_times', type=int, help='Repeat times to calculate training results', default=1)
    parser.add_argument('--include_norm', type=int, help='Whether to include normalization layer in GPU', default=0)
    parser.add_argument('--include_fc', type=int, help='Whether to include fc layer in GPU', default=0)
    # Grey attack arguments
    parser.add_argument('--transferset_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    parser.add_argument('--use_STL10', action='store_true', help='Run on STL10 dataset', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    

    transferset_dir = params['transferset_dir']
    # Set up transferset
    transferset_path = osp.join(transferset_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples
    
    # Set up testset: according to trained_dataset
    dataset_name = params['trained_dataset']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    ori_model, trained_model, trained_acc = load_models(params, num_classes)
    ori_model.to(device)
    trained_model.to(device)

    repeat_times = params['repeat_times']
    for ri in range(repeat_times):
        scaled_model = copy.deepcopy(trained_model)
        scaled_model.to(device)
        recover_model = copy.deepcopy(ori_model)
        recover_model.to(device)

        layer_theta = params['layer_theta']
        include_norm = params['include_norm']
        include_fc = params['include_fc']
        layer_dict = OrderedDict()
        found_conv = False
        last_layer_name = None
        last_size = None

        # for name, param in ori_model.named_parameters():
        #     if 'downsample' in name:
        #         layer_name = '.'.join(name.split('.')[:-2])
        #     else:
        #         layer_name = '.'.join(name.split('.')[:-1])
        #     # Conv layer
        #     if len(list(param.data.size())) == 4:
        #         found_conv = True
        #         layer_dict[layer_name] = [name]
        #         last_layer_name = layer_name
        #         continue
        #     # fc layer
        #     if ('fc' in layer_name) or ('classifier' in layer_name):
        #         if layer_name not in layer_dict.keys():
        #             layer_dict[layer_name] = []
        #         layer_dict[layer_name].append(name)
        #         continue
        #     # Bias
        #     if layer_name in layer_dict.keys():
        #         layer_dict[layer_name].append(name)
        #         continue
        #     # BN
        #     if found_conv:
        #         layer_dict[last_layer_name].append(name)
        #         # BN weight
        #         if not last_size:
        #             last_size = param.data.size()
        #         # BN bias
        #         elif last_size == param.data.size():
        #             last_size = None
        #             found_conv = False
                
        for name, state in ori_model.state_dict().items():
            layer_name = '.'.join(name.split('.')[:-1])

            # Conv layer
            if len(list(state.size())) == 4:
                found_conv = True
                layer_dict[layer_name] = [name]
                last_layer_name = layer_name
                continue
            # fc layer
            if ('fc' in layer_name) or ('classifier' in layer_name):
                if layer_name not in layer_dict.keys():
                    layer_dict[layer_name] = []
                layer_dict[layer_name].append(name)
                continue
            # Bias
            if layer_name in layer_dict.keys() and "bias" in name:
                layer_dict[layer_name].append(name)
                continue
            # BN
            if found_conv:
                
                if "num_batches_tracked" not in name:
                    layer_dict[last_layer_name].append(name)
                else:
                    found_conv = False

        print(len(layer_dict), layer_dict)

        ori_w = 0
        w = ori_w
        unprotect_layername_list = []
        for layer_name in layer_dict.keys():
            if w > 0:
                print(w, layer_name)
            else:
                unprotect_layername_list.append(layer_name)
            w = w - 1

        scaled_count = min( int(len(layer_dict) * layer_theta), len(unprotect_layername_list))
        print(int(len(layer_dict) * layer_theta), len(unprotect_layername_list), scaled_count)
        random.seed(cfg.DEFAULT_SEED)
        if scaled_count > 0:
            random_indices = random.sample(range( len(unprotect_layername_list)-1 ), scaled_count-1)
            random_indices.sort()
            scaled_layername_list = [unprotect_layername_list[i] for i in random_indices]
            scaled_layername_list.append(list(layer_dict.keys())[-1])
        else:
            scaled_layername_list = []
        scaled_name_list = []
        for layer_name in scaled_layername_list:
            scaled_name_list.extend(layer_dict[layer_name])
        print(unprotect_layername_list)
        print(scaled_name_list)
        
        # Get FLOPS
        protect_layername_list = [ln for ln in layer_dict.keys() if ln not in scaled_layername_list]
        protect_flops = 0
        for ln in protect_layername_list:
            protect_flops += trained_model.conv_layer_flops[ln]
        print(protect_layername_list, protect_flops)
        # Get parameter count
        protect_params = 0
        trained_state_dict = trained_model.state_dict()
        for ln in protect_layername_list:
            for name in layer_dict[ln]:
                cur_tensor = trained_state_dict[name]
                protect_params += torch.numel(cur_tensor)
                print(name, torch.numel(cur_tensor))
        print(protect_params)

        # Scale the parameters of scaled_model
        scalar_start = params['scalar_start']
        scalar_end = params['scalar_end']
        scalar_list = []
        scaled_state_dict = copy.deepcopy(scaled_model.state_dict())
        for name in scaled_name_list:
            cur_scalar = random.uniform(scalar_start, scalar_end)
            scalar_list.append(cur_scalar)
            scaled_state_dict[name] = torch.mul(cur_scalar, scaled_state_dict[name])
        scaled_model.load_state_dict(scaled_state_dict)

        # Try to recover the scalar
        ori_state_dict = ori_model.state_dict()
        max_cnt = params['max_cnt']
        est_scalar_list = []
        scalar_results = {}
        scalar_diff = 0
        for i, name in enumerate(scaled_name_list):
            ori_var = torch.var(ori_state_dict[name]).item()
            scaled_var = torch.var(scaled_state_dict[name]).item()
            try:
                est_scalar = math.sqrt(scaled_var / ori_var)
            except ZeroDivisionError:
                est_scalar = 1.0
            est_scalar_list.append(est_scalar)
            scalar_results[name] = [scalar_list[i], est_scalar]
            scalar_diff += abs(scalar_list[i]-est_scalar)
        print(scalar_results)
        try:
            scalar_diff = scalar_diff / scaled_count
        except ZeroDivisionError:
            scalar_diff = 0.0
        
        # Recover the weights using the estimated scalar
        recover_state_dict = copy.deepcopy(recover_model.state_dict())
        for i, name in enumerate(scaled_name_list):
            recover_state_dict[name] = torch.div(scaled_state_dict[name], est_scalar_list[i])

        # Train
        out_dir = params['out_dir']
        budgets = None
        if params['budgets']:
            budgets = [int(b) for b in params['budgets'].split(',')]
        
        for b in budgets:
            np.random.seed(cfg.DEFAULT_SEED)
            torch.manual_seed(cfg.DEFAULT_SEED)
            torch.cuda.manual_seed(cfg.DEFAULT_SEED)
            # Load parameters into the model
            recover_model.load_state_dict(recover_state_dict)

            transferset = samples_to_transferset(transferset_samples, budget=b, transform=transform, \
                use_STL10=params['use_STL10'])
            print()
            print('=> Training at budget = {}'.format(len(transferset)))

            gpu_parameters = [param for name, param in recover_model.named_parameters() if name in scaled_name_list]
            tee_parameters = [param for name, param in recover_model.named_parameters() \
                if (name not in scaled_name_list) and ('fc' not in name) and ('classifier' not in name)]
            fc_parameters = [param for name, param in recover_model.named_parameters() \
                if (name not in scaled_name_list) and ( ('fc' in name) or ('classifier' in name) )]
            param_config = [
                {'params': gpu_parameters, 'lr': params['lr']}, 
                {'params': tee_parameters, 'lr': params['lr']*10}, 
                {'params': fc_parameters, 'lr': 0.03}
            ]
            # recover_optimizer = get_optimizer(recover_model.parameters(), params['optimizer_choice'], **params)
            recover_optimizer = optim.SGD(param_config, params['lr'], momentum=0.5)
            recover_ckpt_suffix = '.recoverscalar.{}.lr{}'.format(b, params['lr'])
            print(recover_ckpt_suffix)
            criterion_train = model_utils.soft_cross_entropy
            model_utils.train_model(recover_model, transferset, out_dir, testset=testset, criterion_train=criterion_train,
                                    checkpoint_suffix=recover_ckpt_suffix, device=device, optimizer=recover_optimizer, **params)


            surrogate_acc, surrogate_fidelity, (clean_top1, adv_top1, adv_sr) = eval_surrogate_model(
                recover_model, trained_model, testset, device
            )
            result = {
                "surrogate_acc": surrogate_acc, 
                "trained_acc": trained_acc, 
                "surrogate_fidelity": surrogate_fidelity,
                "clean_top1": clean_top1,
                "adv_top1": adv_top1,
                "adv_sr": adv_sr,
                "scalar_diff": scalar_diff, 
                "protect_flops": protect_flops, 
                "protect_params": protect_params
            }
            json_name = f"stealing_eval.{ri}.{b}.{params['lr']}.json"
            save_path = osp.join(out_dir, json_name)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=True)
        
        scalar_out_path = osp.join(out_dir, f'scalars.{ri}.json')
        print(scalar_out_path)
        with open(scalar_out_path, 'w') as jf:
            json.dump(scalar_results, jf, indent=True)
        
    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_dir, 'params_train.json')
    print(params_out_path)
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)    
