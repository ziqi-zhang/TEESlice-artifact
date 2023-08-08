#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from datetime import datetime
import json
from collections import defaultdict as dd
from pdb import set_trace as st
import numpy as np
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset

import knockoff.config as cfg
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import knockoff.models.zoo as zoo
from knockoff.victim.blackbox import Blackbox

from knockoff.adv_samples.eval_robustness import advtest, advtest_output

from advertorch.attacks import LinfPGDAttack

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

def myloss(yhat, y):
    return -((yhat[:,0]-y[:,0])**2 + 0.1*((yhat[:,1:]-y[:,1:])**2).mean(1)).mean()

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('--adv_dir', metavar='ADV_DIR', type=str, help='Model name')
    parser.add_argument('--victim_dir', metavar='VICTIM_DIR', type=str, help='Model name')
    parser.add_argument('--pretrained_sup_dir', metavar='VICTIM_DIR', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)

    parser.add_argument("--B", type=float, default=0.2, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument('--budgets', metavar='B', type=str,
                    help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    trainset = dataset(train=True, transform=train_transform)
    testset = dataset(train=False, transform=test_transform)
    num_classes = len(trainset.classes)
    params['num_classes'] = num_classes
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    knockoff_utils.create_dir(params['out_path'])

    def surrogate_attack_penultimate(surrogate, victim, prefix):
        surrogate.attack_backbone = True
        adversary = LinfPGDAttack(
            surrogate, loss_fn=myloss, eps=args.B,
            nb_iter=args.pgd_iter, eps_iter=0.02, 
            rand_init=True, clip_min=-2.42, clip_max=2.75,
            targeted=False)
        clean_top1, adv_top1, adv_sr = advtest(victim, test_loader, adversary, args)
        log = 'Penult {} Clean Top-1: {:.2f} | Adv Top-1: {:.2f} | Attack Success Rate: {:.2f}'.format(prefix, clean_top1, adv_top1, adv_sr)
        print(log)
        with open(osp.join(params['out_path'], f"penult_{prefix}_result.txt"), 'w') as f:
            f.write(log)

    def surrogate_attack_output(surrogate, victim, prefix):
        adversary = LinfPGDAttack(
            surrogate, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.B,
            nb_iter=args.pgd_iter, eps_iter=0.02, 
            rand_init=True, clip_min=-2.42, clip_max=2.75,
            targeted=False)
        clean_top1, adv_top1, adv_sr = advtest_output(victim, test_loader, adversary, args)
        log = 'Output {} Clean Top-1: {:.2f} | Adv Top-1: {:.2f} | Attack Success Rate: {:.2f}'.format(prefix, clean_top1, adv_top1, adv_sr)
        print(log)
        with open(osp.join(params['out_path'], f"output_{prefix}_result.txt"), 'w') as f:
            f.write(log)

    victim_dir = params['victim_dir']
    blackbox = Blackbox.from_modeldir(victim_dir, device)
    victim = blackbox._Blackbox__model
    victim = victim.cuda()
    victim.eval()
    
    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]
    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        if b == 0:
            # Pretrained model attack
            model_name = params['model_arch']
            pretrained = params['pretrained']
            pretrained = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
            pretrained.attack_backbone = True
            pretrained = pretrained.cuda()

            pretrained_sup_dir = params['pretrained_sup_dir']
            blackbox = Blackbox.from_modeldir(pretrained_sup_dir, device)
            pretrained_sup = blackbox._Blackbox__model
            pretrained_sup = pretrained_sup.cuda()

            # pretrained.conv1.load_state_dict(pretrained_sup.conv1.state_dict())
            # pretrained.bn1.load_state_dict(pretrained_sup.bn1.state_dict())
            pretrained.eval()
            surrogate = pretrained

            surrogate_attack_penultimate(pretrained, victim, f"budget{b}")
        else:

            model_name = params['model_arch']
            pretrained = params['pretrained']
            pretrained = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
            pretrained.attack_backbone = True

            adv_dir = params['adv_dir']
            surrogate = Blackbox.load_checkpoint(pretrained, adv_dir, f"checkpoint.{b}.pth.tar")
            surrogate = surrogate.cuda()
            surrogate.eval()

            surrogate_attack_penultimate(pretrained, victim, f"budget{b}")
            surrogate_attack_output(pretrained, victim, f"budget{b}")


        




if __name__ == '__main__':
    main()
