#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np
from pdb import set_trace as st
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models

import knockoff.config as cfg
import knockoff.utils.utils as knockoff_utils

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

featloss = torch.nn.MSELoss()
# Used to matching features
def record_act(self, input, output):
    self.out = output

def train_step_feat(model, teacher, train_loader, criterion, optimizer, epoch, device, feat_lmda, reg_layers, log_interval=10, writer=None):
    model.train()
    teacher.eval()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         if hasattr(module, 'weight'):
    #             module.weight.requires_grad_(False)
    #         if hasattr(module, 'bias'):
    #             module.bias.requires_grad_(False)
    #         module.track_running_stats = False
            # module.eval()
    

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        loss = criterion(outputs, targets)

        regloss = 0
        for (module, t_module) in reg_layers:
            regloss += featloss(module.out, t_module.out.detach())
        regloss = feat_lmda * regloss
        loss += regloss

        loss.backward()
        optimizer.step()

        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc


def test_step(model, teacher, test_loader, criterion, device, feat_lmda, reg_layers, epoch=0., silent=False, writer=None):
    model.eval()
    test_loss = 0.
    cls_loss = 0.
    feat_loss = 0.
    correct = 0
    total = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            teacher_outputs = teacher(inputs)
            loss = criterion(outputs, targets)

            regloss = 0
            for (module, t_module) in reg_layers:
                regloss += featloss(module.out, t_module.out.detach())
            regloss = feat_lmda * regloss
            loss += regloss

            nclasses = outputs.size(1)

            test_loss += loss.item()
            feat_loss += regloss.item()
            cls_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    test_loss /= total
    feat_loss /= total
    cls_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tFeat: {:.6f}\tCls: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, feat_loss, cls_loss, acc,
                                                                             correct, total))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc


def train_model_feat(model, teacher, trainset, out_path, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, feat_lmda=1, **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None

    if weighted_loss:
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss = -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            wf.write('\t'.join(columns) + '\n')

    reg_layers = []
    # for (name, module), (t_name, t_module) in zip(model.named_modules(), teacher.named_modules()):
    #     # print(name)
    #     if isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.MaxPool2d):
    #         print(name)
    #         module.register_forward_hook(record_act)
    #         t_module.register_forward_hook(record_act)
    #         reg_layers.append((module, t_module))
    model.layer1.register_forward_hook(record_act)
    teacher.layer1.register_forward_hook(record_act)
    reg_layers.append((model.layer1, teacher.layer1))

    model.layer2.register_forward_hook(record_act)
    teacher.layer2.register_forward_hook(record_act)
    reg_layers.append((model.layer2, teacher.layer2))

    model.layer3.register_forward_hook(record_act)
    teacher.layer3.register_forward_hook(record_act)
    reg_layers.append((model.layer3, teacher.layer3))

    model.layer4.register_forward_hook(record_act)
    teacher.layer4.register_forward_hook(record_act)
    reg_layers.append((model.layer4, teacher.layer4))

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_step_feat(model, teacher, train_loader, criterion_train, optimizer, epoch, device,
                                           log_interval=log_interval, feat_lmda=feat_lmda, reg_layers=reg_layers)
        scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if test_loader is not None:
            test_loss, test_acc = test_step(
                model, teacher, test_loader, criterion_test, device, 
                epoch=epoch, feat_lmda=feat_lmda, reg_layers=reg_layers
            )
            best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint
        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

        # Log
        with open(log_path, 'a') as af:
            train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model
