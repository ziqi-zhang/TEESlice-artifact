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
from knockoff.nettailor_face.dataset_info import *
from sklearn.model_selection import train_test_split

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


class UTKFace(torch.utils.data.Dataset):
    # def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
    def __init__(self, train=True, transform=None, target_transform=None, attr = "age")-> None:
        # attr = ["race", "gender", "age"]
        # attr = "age"

        # self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.files = os.listdir(self.root+'/UTKFace/processed/')
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]
        if attr == "gender":
            self.classes = list(range(2))
        elif attr == "race":
            self.classes = list(range(4))
        elif attr == "age":
            self.classes = list(range(117))
        else:
            raise RuntimeError
        

        self.raw_lines = []
        # self.samples = []
        for txt_file in self.files:
            with open(self.root+'/UTKFace/processed/' + txt_file, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4:
                        continue
                    if attrs[1] == "":
                        continue
                    self.raw_lines.append(image_name+'jpg')
                    # self.samples.append(

                    # )

        train_lines, test_lines = train_test_split(self.raw_lines, test_size=0.1, random_state=37)
        if train:
            self.lines = train_lines
        else:
            self.lines = test_lines
        print(self.raw_lines[:10])
        print("UTKFace raw samples: ", len(self.raw_lines))
        # exit()

    def getitem_to_numpy(self, index):
        image_path = os.path.join(self.root+'/UTKFace/UTKface_aligned_cropped/UTKFace', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        image = (image.numpy()*255).astype(np.uint8).transpose(1,2,0)
        return image


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        
        attrs = self.lines[index].split('_')
        # print(self.lines[index], attrs, attrs[0], attrs[1], attrs[2])

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root+'/UTKFace/UTKface_aligned_cropped/UTKFace', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

class UTKFaceAge(UTKFace):
    # def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
    def __init__(self, train=True, transform=None, target_transform=None)-> None:
        super().__init__(train, transform, target_transform, "age")

class UTKFaceRace(UTKFace):
    # def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
    def __init__(self, train=True, transform=None, target_transform=None)-> None:
        super().__init__(train, transform, target_transform, "race")


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
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize( **NORMALIZE_DICT[dataset_name.lower()] ),
        ])
        transform = train_transform if mode=="train" else val_transform
        dataset = STL10(train=mode=="train", transform=transform, download=True)
        # test = STL10('data/cifar10', train=False, transform=transform, download=True)
        dataset.num_classes = 10

    elif dataset_name.lower() == 'utkfacerace':
        num_classes = 4
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT[dataset_name.lower()]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize( **NORMALIZE_DICT[dataset_name.lower()] ),
        ])
        transform = train_transform if mode=="train" else val_transform
        dataset = UTKFaceRace(train=mode=="train", transform=transform, )
        # test = STL10('data/cifar10', train=False, transform=transform, download=True)
        dataset.num_classes = 4

    
    print(f"Get {mode} dataset size {len(dataset)}")
    return num_classes, dataset

def get_dataloader(dataset, batch_size=1, shuffle=True, mode='train', num_workers=4, val_ratio=0.1):

    num_classes, dataset = get_model_dataset(dataset, mode, val_ratio=val_ratio)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader

