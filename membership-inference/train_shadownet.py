import os, sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os.path as osp
import json

from doctor.meminf_whitebox import *
from doctor.meminf_blackbox import *
from doctor.meminf_shadow import *
from doctor.meminf_whitebox_feature import *
from doctor.modinv import *
from doctor.attrinf import *
from doctor.modsteal import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *
from log_utils import *
from distill import *
from mem_attack import test_meminf_full, test_meminf_no_train

from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset

import knockoff_train
from utils_shadownet import *

import gol
gol._init()
gol.set_value("debug", False)

def load_target_model(target_model, args):
    target_path = osp.join(args.victim_dir, f"target.pth")
    target_model = target_model.cuda()
    print(f"Load model from {target_path}")
    target_model.load_state_dict(torch.load(target_path))
    target_model.eval()
    return target_model

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
        self.ground_truths = [self.samples[i][2] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target, gt = self.data[index], self.targets[index], self.ground_truths[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.dtype)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, gt

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

def load_target_model(target_model, args):
    target_path = osp.join(args.victim_dir, f"target.pth")
    target_model = target_model.cuda()
    print(f"Load model from {target_path}")
    target_model.load_state_dict(torch.load(target_path))
    target_model.eval()
    return target_model

def test_model(model, testset, args):
    model = load_target_model(model, args)

    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=0)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, soft_targets, targets in dataloader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, soft_targets, targets = inputs.to(device), soft_targets.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
    acc = 1.*correct/total
    print(acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model')
    parser.add_argument('--victim_dir', type=str)
    parser.add_argument('--shadow_model_dir', type=str)
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    parser.add_argument('--trasnferset_budgets', type=int, default=1000)
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--shadow-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--remain-lr', type=float, default=1e-2)
    parser.add_argument('--update-lr', type=float, default=1e-1)
    parser.add_argument('--shadow-lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', action="store_true", default=False)
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(17)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    assert args.pretrained
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, pretrained_model = prepare_dataset(
        args.dataset.lower(), args.model_arch, pretrained=args.pretrained
    )

    if isinstance(target_train.dataset, torch.utils.data.dataset.ConcatDataset):
        transform = target_train.dataset.datasets[0].transform
    elif isinstance(target_train.dataset, UTKFaceDataset):
        transform = target_train.dataset.transform
    else:
        raise NotImplementedError

    
    transferset_path = osp.join(args.out_path, "transferset.pickle")
    if not os.path.exists(transferset_path):
        raise RuntimeError
    
    # ----------- Set up transferset
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i, gt_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            # new_transferset_samples.append((x_i, argmax_k, gt_i))
            # st()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot, gt_i))
        transferset_samples = new_transferset_samples

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]
    

    for b in budgets:
        np.random.seed(37)
        torch.manual_seed(37)
        torch.cuda.manual_seed(37)

        transferset = samples_to_transferset(transferset_samples, budget=b, transform=transform)
        print()
        print('=> Training at budget = {}'.format(len(transferset)))
        
        pretrained = copy.deepcopy(target_model)
        model = copy.deepcopy(pretrained)
        model = load_target_model(model, args)
        
        ori_model, trained_model = pretrained, model
        ori_model.to(device)
        trained_model.to(device)
        shuffle_model = copy.deepcopy(trained_model)
        shuffle_model.to(device)
        recover_model = copy.deepcopy(trained_model)
        recover_model.to(device)

        name_list = []
        size_list = []
        print(params['model_arch'])
        for name, param in ori_model.named_parameters():
            print(name, param.data.size())
            if param.requires_grad and (len(list(param.data.size())) == 4) and ('downsample' not in name) \
                and (name != 'conv1.weight') and (name != 'features.0.weight') and (name != 'layer0.conv1.weight') \
                and (name != 'conv1.0.weight'): 
                # print(name, param.data, trained_model.state_dict()[name])
                ori_layer = ori_model.state_dict()[name]
                kernel_size = ori_layer.size()[1] * ori_layer.size()[2] * ori_layer.size()[3]
                ori_n_outchannels = ori_layer.size()[0]
                if ori_n_outchannels > kernel_size:
                    continue
                name_list.append(name)
                size_list.append(param.data.size())
                

        # print(name_list)
        # print(size_list)

        # shuffle kernels in shuffle_model without recovering
        # shuffle kernels in recover_model with recovery
        r = 1.2
        t = 10.0
        # [NOTE]: use deepcopy here to make the parameters independent
        # we will train the recover_model with different budgets
        # and previous training should not affect later training
        shuffle_state_dict = copy.deepcopy(shuffle_model.state_dict())
        recover_state_dict = copy.deepcopy(recover_model.state_dict())
        recovery_results = {}
        for kernel_name in name_list:
            if kernel_name == name_list[0]:
                nbins = 20
                nrows = 4
                ncols = 4
                # plot_hist_box(params, ori_model, f'pretrained_{kernel_name}', kernel_name, nbins, nrows, ncols)
                # plot_hist_box(params, trained_model, f'finetuned_{kernel_name}', kernel_name, nbins, nrows, ncols)
            ori_layer_flatten, trained_layer_flatten, new_layer_flatten, frand_indices, shuffle_indices = \
                shadownet_layer(ori_model, trained_model, kernel_name, r, t)
            print(kernel_name, trained_layer_flatten.shape, new_layer_flatten.shape)
            shuffle_indices, shuffle_guess_indices = \
                attack_shadownet_layer_nolamb(ori_layer_flatten, trained_layer_flatten, new_layer_flatten, \
                r, t, frand_indices, shuffle_indices)
            
            # Record recovery results
            ori_n_outchannels = trained_layer_flatten.shape[0]
            equal_arr = (shuffle_indices == shuffle_guess_indices)
            recovery_results[kernel_name] = [int(equal_arr.sum()), int(ori_n_outchannels)]

            # shuffle weights
            axis = 0
            shuffle_state_dict[kernel_name] = shuffle_state_dict[kernel_name][shuffle_indices, :,:,:]
            # recover weights
            recover_state_dict[kernel_name] = recover_state_dict[kernel_name][shuffle_indices, :,:,:]
            for ki in range(ori_n_outchannels):
                recover_state_dict[kernel_name][shuffle_guess_indices[ki], :,:,:] = \
                    shuffle_state_dict[kernel_name][ki, :,:,:]
        
        
        adv_model_dir = osp.join(args.out_path, f"shadownet_{b}")
        # optimizer = get_optimizer(param_config, params['optimizer_choice'], **params)
        optimizer = optim.SGD(recover_model.parameters(), args.lr, momentum=args.momentum)
        print(params)
        
        checkpoint_suffix = '.shadownet.{}'.format( b)
        criterion_train = knockoff_train.soft_cross_entropy
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        knockoff_train.train_model(
            model, transferset, adv_model_dir, batch_size=args.batch_size, epochs=args.epochs,
            testset=shadow_test, criterion_train=criterion_train,
            checkpoint_suffix=checkpoint_suffix, device=device, 
            optimizer=optimizer, scheduler=scheduler)
        
        target_path = osp.join(adv_model_dir, "target.pth")
        torch.save(model.state_dict(), target_path)
        
        
        test_meminf_no_train(
            adv_model_dir, adv_model_dir, args.shadow_model_dir,
            device, num_classes, target_train, target_test, shadow_train, shadow_test, 
            target_model, shadow_model, args, 
        )
        
        del model
