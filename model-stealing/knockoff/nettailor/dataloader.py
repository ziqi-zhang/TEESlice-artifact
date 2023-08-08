import os
import torch
import pandas
import torchvision
torch.manual_seed(0)
import torch.nn as nn
from torchvision import datasets
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.datasets import STL10 as RawSTL10
import os.path as osp

from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
from pdb import set_trace as st
from knockoff.nettailor.dataset_info import *

def pil_loader(path):
    return Image.open(path).convert('RGB')
class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, loader=pil_loader, target_transform=None):
        super(MyImageFolder, self).__init__(root=root, transform=transform, loader=pil_loader, target_transform=target_transform)
        self.id2img = {path.replace(self.root, ''): path for path, _ in self.imgs}

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        assert isinstance(target, int)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        iid = path.replace(self.root, '')
        return img, target

class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, data_root, train, transform=None, download=True):
        super(CIFAR10Dataset, self).__init__(root=data_root, train=train, transform=transform, download=download)
        self.sample_subset = False
        self.num_classes = 10
        
    def set_subset(self, ratio):
        self.sample_subset = True
        num_sample = int(ratio * len(self.data))
        raw_idxs = list(range(len(self.data)))
        self.sample_idxs = np.random.choice(raw_idxs, size=num_sample)
        
    def __len__(self):
        if self.sample_subset:
            return len(self.sample_idxs)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.sample_subset:
            img, target = self.data[self.sample_idxs[index]], int(self.targets[self.sample_idxs[index]])
            real_index = self.sample_idxs[index]
        else:
            img, target = self.data[index], int(self.targets[index])
            real_index = index
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # iid = '{}_{:06d}'.format(self.split, index)
        return img, target

class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self, data_root, train, transform=None, download=True):
        super(CIFAR100Dataset, self).__init__(root=data_root, train=train, transform=transform, download=download)
        self.sample_subset = False
        self.num_classes = 100
        
    def set_subset(self, ratio):
        self.sample_subset = True
        num_sample = int(ratio * len(self.data))
        raw_idxs = list(range(len(self.data)))
        self.sample_idxs = np.random.choice(raw_idxs, size=num_sample)
        
    def __len__(self):
        if self.sample_subset:
            return len(self.sample_idxs)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.sample_subset:
            img, target = self.data[self.sample_idxs[index]], int(self.targets[self.sample_idxs[index]])
            real_index = self.sample_idxs[index]
        else:
            img, target = self.data[index], int(self.targets[index])
            real_index = index
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # iid = '{}_{:06d}'.format(self.split, index)
        return img, target

class STL10(RawSTL10):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('data', 'stl10')
        if train:
            split = 'train'
        else:
            split = 'test'
        super().__init__(root, split, None, transform, target_transform, True)



def prepare_dataset(dataset, attr, root, model="resnet18"):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, attr=attr, root=root, model=model)
    length = len(dataset)
    each_length = length//4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model

def prepare_proxy_dataset(dataset, attr, root):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, attr=attr, root=root)

    return num_classes, dataset, target_model, shadow_model


def get_model_dataset(dataset_name, mode, val_ratio=0.1):
    random_seed = 7
    torch.manual_seed(random_seed) 
    torch.cuda.manual_seed(random_seed) 
    np.random.seed(random_seed) 
    random.seed(random_seed)
    
    if dataset_name.lower() == 'cifar100':
        num_classes = 100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name.lower()]),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( **NORMALIZE_DICT[dataset_name.lower()] ),
        ])
        transform = train_transform if mode=="train" else val_transform
        dataset = CIFAR100Dataset('data/cifar100', train=mode=="train", transform=transform, download=True)
        # test = CIFAR100Dataset('data/cifar100', train=False, transform=transform, download=True)
        # dataset = train + test
    elif dataset_name.lower() == 'cifar10':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name.lower()]),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( **NORMALIZE_DICT[dataset_name.lower()] ),
        ])
        transform = train_transform if mode=="train" else val_transform
        dataset = CIFAR10Dataset('data/cifar10', train=mode=="train", transform=transform, download=True)
        # test = CIFAR10Dataset('data/cifar10', train=False, transform=transform, download=True)

    elif dataset_name.lower() == 'stl10':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name.lower()]),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( **NORMALIZE_DICT[dataset_name.lower()] ),
        ])
        transform = train_transform if mode=="train" else val_transform
        dataset = STL10(train=mode=="train", transform=transform, download=True)
        # test = STL10('data/cifar10', train=False, transform=transform, download=True)
        dataset.num_classes = 10

    
    print(f"Get {mode} dataset size {len(dataset)}")
    return num_classes, dataset

def get_dataloader(dataset, batch_size=1, shuffle=True, mode='train', num_workers=4, val_ratio=0.1):

    num_classes, dataset = get_model_dataset(dataset, mode, val_ratio=val_ratio)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader

