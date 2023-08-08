import os
import torch
import pandas
import torchvision
torch.manual_seed(0)
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np
import random

from resnet32 import create_model

from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
from pdb import set_trace as st
import medmnist

class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*2*2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_dataloader(dataset, batch_size=1, shuffle=True, mode='train', num_workers=4):
    
    train = get_dataset(dataset, "train")
    val = get_dataset(dataset, "val")
    test = get_dataset(dataset, "test")
    dataset = train + val + test
    num_classes = train.num_classes

    length = len(dataset)
    
    random_seed = 7
    torch.manual_seed(random_seed) 
    torch.cuda.manual_seed(random_seed) 
    np.random.seed(random_seed) 
    random.seed(random_seed)
    # split_length = length // (9*4+1)
    # target_train, target_test, target_val, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [split_length*9, split_length*9, split_length, split_length*9, split_length*9, len(dataset)-(split_length*37)])
    # split_length = length // (2*4+1)
    # target_train, target_test, target_val, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [split_length*2, split_length*2, split_length, split_length*2, split_length*2, len(dataset)-(split_length*9)])
    split_length = length // 5
    target_train, target_test, target_val, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [split_length, split_length, split_length, split_length, split_length, len(dataset)-(split_length*5)])
    target_train.num_classes = num_classes
    target_test.num_classes = num_classes
    target_val.num_classes = num_classes

    if mode == 'train':
        loader = torch.utils.data.DataLoader(target_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(target_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    
    return loader

def prepare_dataset(dataset, model="resnet18"):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, model=model)
    
    length = len(dataset)
    
    random_seed = 7
    torch.manual_seed(random_seed) 
    torch.cuda.manual_seed(random_seed) 
    np.random.seed(random_seed) 
    random.seed(random_seed)
    split_length = length // 5
    target_train, target_test, target_val, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [split_length, split_length, split_length, split_length, split_length, len(dataset)-(split_length*5)])
    
    return num_classes, target_train, target_test, target_val, shadow_train, shadow_test, target_model, shadow_model

def prepare_proxy_dataset(dataset, attr, root):
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, attr=attr, root=root)

    return num_classes, dataset, target_model, shadow_model


def get_model_dataset(dataset_name, model="resnet18"):
    
    if 'MNIST' in dataset_name and dataset_name not in ['mnist']:
        # Data augmentation
        image_size = 32
        crop_size = 32
        input_channel = 3

        transform = transforms.Compose([
            transforms.Scale(image_size),
            # transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


        dataset = train+val+test
        dataset.num_classes = len(train.info['label'])
        num_classes = dataset.num_classes


    if model == "cnn":
        if isinstance(num_classes, int):
            target_model = CNN(input_channel=input_channel, num_classes=num_classes)
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes)
        else:
            target_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
    elif model.startswith("resnet") :
        if isinstance(num_classes, int):
            target_model = create_model(model, num_classes=num_classes)
            shadow_model = create_model(model, num_classes=num_classes)
        else:
            target_model = create_model(model, num_classes=num_classes[0])
            shadow_model = create_model(model, num_classes=num_classes[0])
        
        pretrain_path = f"medmnist_model/cifar10/{model}/checkpoint.pth.tar"
        
        ckp = torch.load(pretrain_path)['state_dict']
        state_dict = target_model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        target_model.load_state_dict(ckp)
        
        state_dict = shadow_model.state_dict()
        if ckp['fc.weight'].size(0) != state_dict['fc.weight'].size(0):
            ckp['fc.weight'] = state_dict['fc.weight']
            ckp['fc.bias'] = state_dict['fc.bias']
        shadow_model.load_state_dict(ckp)
    else:
        raise NotImplementedError

    return num_classes, dataset, target_model, shadow_model