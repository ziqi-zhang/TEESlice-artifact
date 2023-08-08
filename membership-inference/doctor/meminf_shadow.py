import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
np.set_printoptions(threshold=np.inf)
from pdb import set_trace as st
from log_utils import *
import os.path as osp

# from opacus import PrivacyEngine
from torch.optim import lr_scheduler
# from opacus.utils import module_modification
from sklearn.metrics import f1_score, roc_auc_score
# from opacus.dp_model_inspector import DPModelInspector

import gol

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class shadow():
    def __init__(self, trainloader, testloader, model, device, batch_size, loss, optimizer, logger, epochs):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs

        self.criterion = loss
        self.optimizer = optimizer
        self.logger = logger

        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [int(epochs*0.6), int(epochs*0.8)], 0.5)
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [int(epochs*0.5), int(epochs*1)], 0.1)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [int(epochs*0.5), int(epochs*0.75)], 0.1)


    # Training
    def train(self):
        self.model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if gol.get_value("debug"):
                break

        self.scheduler.step()

        self.logger.add_line( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/(batch_idx+1)))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if gol.get_value("debug"):
                    break

            self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total


def train_shadow_model(
    PATH, device, shadow_model, train_loader, test_loader, 
    batch_size, loss, optimizer, logger, args
):
    epochs = args.shadow_epochs
    model = shadow(
        train_loader, test_loader, shadow_model, device, 
        batch_size, loss, optimizer, logger, epochs
    )
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("shadow training")

        acc_train = model.train()
        logger.add_line("shadow testing")
        acc_test = model.test()


        overfitting = round(acc_train - acc_test, 6)

        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    # FILE_PATH = PATH + "shadow.pth"
    FILE_PATH = os.path.join(PATH, "shadow.pth")
    model.saveModel(FILE_PATH)
    logger.add_line(f"saved shadow model to {FILE_PATH}!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting


def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)


    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader
