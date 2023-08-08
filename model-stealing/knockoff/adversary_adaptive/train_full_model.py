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
import copy
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo
from knockoff.victim.blackbox import Blackbox

# from knockoff.nettailor import teacher_resnet 
from knockoff.adversary_adaptive import student_resnet
# from knockoff.nettailor import dataloader as dataloaders
# from knockoff.nettailor import dataset_info
from knockoff.adversary_adaptive import proj_utils



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
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
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


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    # parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    # parser.add_argument('victim_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('victim_model_dir', metavar='DIR', type=str)
    parser.add_argument('nettailor_model_dir', type=str)
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
    parser.add_argument('--backbone-lr', type=float, default=0.01)
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
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    # parser.add_argument('--backbone', metavar='BACKBONE', default='resnet34',
    #                 help='backbone model architecture: ' + ' (default: resnet34)')
    # parser.add_argument('--max-skip', default=3, type=int)

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

    victim_dir = params['victim_model_dir']
    blackbox = Blackbox.from_modeldir(victim_dir, device)
    victim_model = blackbox._Blackbox__model
    victim_model = victim_model.to(device)

    nettailor_model = student_resnet.create_model(
        num_classes=num_classes, 
        max_skip=3,
        backbone='resnet18',
    )
    print("\n"+"="*30+"   Checkpoint   "+"="*30)
    print("Loading checkpoint from: " + args.nettailor_model_dir)
    proj_utils.load_checkpoint(nettailor_model, model_dir=args.nettailor_model_dir)
    nettailor_model = nettailor_model.to(device)

    model = copy.deepcopy(nettailor_model)


    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]

    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        model.init_proxy_weights()

        # parameters = [
        #     {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'proxies' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'alphas_params' in n], 'lr': args.lr, 'weight_decay': 0.0},
        #     {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'ends_bn' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        # ]
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=0.0004)

        transferset = samples_to_transferset(transferset_samples, budget=b, transform=transform)
        print()
        print('=> Training at budget = {}'.format(len(transferset)))

        # optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], **params)
        # print(params)

        checkpoint_suffix = '.{}'.format(b)
        criterion_train = model_utils.soft_cross_entropy
        model_utils.train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, **params)

        # eval model stealing
        from knockoff.adversary.eval_stealing_graybox import eval_surrogate_model
        # ckpt_name = f"checkpoint.{b}.pth.tar"
        # path = osp.join(args.model_dir, ckpt_name)
        # surrogate_model = Blackbox.load_checkpoint(model, args.model_dir, ckpt_name)
        # surrogate_model = surrogate_model.to(device)

        surrogate_acc, surrogate_fidelity, (clean_top1, adv_top1, adv_sr) = eval_surrogate_model(
            model, victim_model, testset, device
        )
        stats = model.stats()
        path = osp.join(args.model_dir, "stats.txt")
        with open(path, "w") as f:
            f.write(stats)

        result = {
            "surrogate_acc": surrogate_acc, 
            "surrogate_fidelity": surrogate_fidelity,
            "clean_top1": clean_top1,
            "adv_top1": adv_top1,
            "adv_sr": adv_sr,
            # "param_protected": param_protected,
            # "param_total": param_total,
            # "flop_protected": flop_protected,
            # "flop_total": flop_total
        }

        json_name = f"stealing_eval.{b}.json"
        save_path = osp.join(args.model_dir, json_name)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=True)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
