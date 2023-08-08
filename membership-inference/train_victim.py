import os, sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os.path as osp

# from doctor.meminf import *
from doctor.modinv import *
from doctor.attrinf import *
from doctor.modsteal import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *
from log_utils import *
from distill import *

import gol
gol._init()
gol.set_value("debug", False)


def train_model(PATH, device, train_set, test_set, model, args):
    debug = gol.get_value("debug")
    
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=PATH)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    
    model = model_training(train_loader, test_loader, model, device, logger, args.epochs, args.lr)
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, )
    acc_train = 0
    acc_test = 0

    for i in range(args.epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target training")

        acc_train = model.train()
        logger.add_line("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    # FILE_PATH = PATH + "_target.pth"
    FILE_PATH = osp.join(PATH, "target.pth")
    model.saveModel(FILE_PATH)
    logger.add_line(f"Saved target model to {FILE_PATH}!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.1)')
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

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, pretrained_model = prepare_dataset(
        args.dataset.lower(), args.model_arch, pretrained=args.pretrained
    )
    train_model(args.out_path, device, target_train, target_test, target_model, args)
    # test_meminf(
    #     TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    #     target_model, shadow_model, "target"
    # )



    
    # TARGET_DIR = target_root + name + "/"
    # if not os.path.exists(TARGET_DIR):
    #     os.makedirs(TARGET_DIR)
    # TARGET_PATH = TARGET_DIR + name    

    # num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(name, attr, root, "resnet18", pretrained=pretrained)
    # train_model(TARGET_PATH, device, target_train, target_test, target_model)
    # test_meminf(
    #     TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    #     target_model, shadow_model, "target"
    # )