import os, sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from pdb import set_trace as st

from doctor.meminf import *
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

def graybox_layer_model_stealing(
    PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    target_model, shadow_model, pretrained_model, args, budget, target_name = "target"
):
    LOG_PATH = PATH + f"_test_meminf_{target_name}/"
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    target_path = PATH + f"_{target_name}.pth"
    target_model.load_state_dict(torch.load(target_path))
    shadow_path = PATH + "_" + "shadow.pth"
    
    # train_subset = 
    st()
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('graybox_model_dir', metavar='DIR', type=str)
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

    torch.manual_seed(17)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']


    pretrained = True
    target_root = "./trained_model_mem_pretrained/"
    
    # pretrained = False
    # target_root = "./trained_model_mem_scratch/"
    
    if args.dataset.lower() == "stl10":
    
        name = "stl10"
        attr = "attr"
    else:
        raise NotImplementedError
    

    TARGET_DIR = target_root + name + "/"
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    TARGET_PATH = TARGET_DIR + name    
    
    

    
    # train_model(TARGET_PATH, device, target_train, target_test, target_model)
    
    budgets = [int(b) for b in params['budgets'].split(',')]
    for budget in budgets:
        num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, pretrained_model = prepare_dataset(name, attr, root, args.model_arch, pretrained=pretrained)

        graybox_layer_model_stealing(
            TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
            target_model, shadow_model, pretrained_model, args, budget, "target"
        )
