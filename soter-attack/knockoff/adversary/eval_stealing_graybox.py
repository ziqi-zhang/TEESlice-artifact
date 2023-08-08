#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime
from pdb import set_trace as st

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo
from knockoff.victim.blackbox import Blackbox
from advertorch.attacks import LinfPGDAttack

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

def eval_model(model, testset, device):
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            nclasses = outputs.size(1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc

def eval_surrogate_fidelity(surrogate, victim, testset, device):
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    surrogate.eval()
    victim.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            surrogate_outputs = surrogate(inputs)
            _, surrogate_predicted = surrogate_outputs.max(1)
            victim_outputs = victim(inputs)
            _, victim_predicted = victim_outputs.max(1)
            total += targets.size(0)
            correct += surrogate_predicted.eq(victim_predicted).sum().item()

    fidelity = 100. * correct / total
    return fidelity

def advtest_output(model, loader, adversary):
    model.eval()

    total_ce = 0
    total = 0
    top1 = 0

    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for i, (batch, label) in enumerate(loader):
        batch, label = batch.to('cuda'), label.to('cuda')
        total += batch.size(0)
        out_clean = model(batch)
        _, pred_clean = out_clean.max(dim=1)
        
        advbatch = adversary.perturb(batch, pred_clean)

        out_adv = model(advbatch)
        _, pred_adv = out_adv.max(dim=1)
        

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

        print('{}/{}...'.format(i+1, len(loader)))
        if i > 10:
            break
    return float(top1_clean)/total*100, float(top1_adv)/total*100, float(adv_trial-adv_success) / adv_trial *100


def eval_adversarial_transfer(surrogate, victim, testset, device):
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    surrogate.eval()
    victim.eval()

    adversary = LinfPGDAttack(
        surrogate, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2,
        nb_iter=20, eps_iter=0.02, 
        rand_init=True, clip_min=-2.42, clip_max=2.75,
        targeted=False)
    clean_top1, adv_top1, adv_sr = advtest_output(victim, test_loader, adversary)
    print(f"clean top1 {clean_top1}, adv top1 {adv_top1}, adv sr {adv_sr}")
    return clean_top1, adv_top1, adv_sr

def eval_surrogate_model(surrogate, victim, testset, device):
    surrogate_acc = eval_model(surrogate, testset, device)
    surrogate_fidelity = eval_surrogate_fidelity(surrogate, victim, testset, device)
    clean_top1, adv_top1, adv_sr = eval_adversarial_transfer(surrogate, victim, testset, device)

    return surrogate_acc, surrogate_fidelity, (clean_top1, adv_top1, adv_sr)

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('victim_model_dir', metavar='DIR', type=str)
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--remain-lr', type=float, default=1e-2)
    parser.add_argument('--update-lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=0)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--graybox-mode', type=str, choices=['block_deep', 'block_shallow'])
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']

    # ----------- Set up transferset
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
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

    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    pretrained = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    pretrained = pretrained.to(device)

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        victim_dir = params['victim_model_dir']
        blackbox = Blackbox.from_modeldir(victim_dir, device)
        victim_model = blackbox._Blackbox__model

        # print("Modules:")
        # for name, module in graybox_model.named_modules():
        #     print(name)
        # print("Parameters:")
        # for name, module in graybox_model.named_parameters():
        #     print(name)

        for num_layer in range(pretrained.total_blocks+1):
        # for num_layer in [graybox_model.total_blocks]:
        # for num_layer in [3]:
            ckpt_name = f"checkpoint.{args.graybox_mode}.{num_layer}.{b}.pth.tar"
            path = osp.join(args.model_dir, ckpt_name)
            surrogate_model = Blackbox.load_checkpoint(pretrained, args.model_dir, ckpt_name)

            surrogate_acc, surrogate_fidelity, (clean_top1, adv_top1, adv_sr) = eval_surrogate_model(
                surrogate_model, victim_model, testset, device
            )

            result = {
                "surrogate_acc": surrogate_acc, 
                "surrogate_fidelity": surrogate_fidelity,
                "clean_top1": clean_top1,
                "adv_top1": adv_top1,
                "adv_sr": adv_sr,
            }
            json_name = f"stealing_eval.{args.graybox_mode}.{num_layer}.{b}.json"
            save_path = osp.join(args.model_dir, json_name)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=True)


    #         checkpoint_suffix = '.{}.{}.{}'.format(args.graybox_mode, num_layer, b)
    #         criterion_train = model_utils.soft_cross_entropy
    #         model_utils.train_model(graybox_model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
    #                                 checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)

    # # Store arguments
    # params['created_on'] = str(datetime.now())
    # params_out_path = osp.join(model_dir, 'params_train.json')
    # with open(params_out_path, 'w') as jf:
    #     json.dump(params, jf, indent=True)



if __name__ == '__main__':
    main()
