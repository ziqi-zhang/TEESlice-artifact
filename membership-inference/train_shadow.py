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
from mem_attack import *

import gol
gol._init()
gol.set_value("debug", False)


# def test_meminf_full(
#     save_dir, target_model_dir, shadow_model_dir, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
#     target_model, shadow_model, args,
# ):
#     # LOG_PATH = PATH + f"_test_meminf_{target_name}/"
#     LOG_PATH = save_dir
#     if not os.path.exists(LOG_PATH):
#         os.makedirs(LOG_PATH)
        
#     shadow_model_path = osp.join(shadow_model_dir, "shadow.pth")
    

#     if os.path.exists(shadow_model_path):
#         print(f"Shadow model exists in {shadow_model_path}")
#     else:
#         print("Training shadow model")

#         shadow_trainloader = torch.utils.data.DataLoader(
#             shadow_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
#         shadow_testloader = torch.utils.data.DataLoader(
#             shadow_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

#         loss = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(shadow_model.parameters(), lr=args.shadow_lr, momentum=0.9, weight_decay=5e-4)
#         shadow_model_logger = Logger(log2file=True, mode="shadow_model", path=LOG_PATH)
#         train_shadow_model(
#             shadow_model_dir, device, shadow_model, shadow_trainloader, shadow_testloader, 
#             args.batch_size, loss, optimizer, shadow_model_logger, args
#         )
#         print("Shadow model training finished")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model')
    parser.add_argument('--victim_dir', type=str)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--shadow-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.1)')
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

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, pretrained_model = prepare_dataset(
        args.dataset.lower(), args.model_arch, pretrained=args.pretrained
    )
    # train_model(args.out_path, device, target_train, target_test, target_model, args)
    # test_meminf_full(
    #     args.out_path, args.victim_dir, args.out_path,
    #     device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    #     target_model, shadow_model, args
    # )
    test_meminf_full(
        args.out_path, args.victim_dir, args.out_path,
        device, num_classes, target_train, target_test, shadow_train, shadow_test, 
        target_model, shadow_model, args
    )



    
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