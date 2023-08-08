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

def compute_generalization_gap(model, model_path, mem_set, nonmem_set, device):
    print("compute_generalization_gap")
    model = model.to(device)
    print(f"Load model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mem_loader = torch.utils.data.DataLoader(
        mem_set, batch_size=64, shuffle=True, num_workers=8)
    nonmem_loader = torch.utils.data.DataLoader(
        nonmem_set, batch_size=64, shuffle=True, num_workers=8)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in mem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
    mem_acc = 1.*correct/total
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in nonmem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
    nonmem_acc = 1.*correct/total
    
    gap = mem_acc - nonmem_acc
    print(f"Generalization gap: member acc {mem_acc:.2f}, non member acc {nonmem_acc:.2f}, gap {gap:.2f}")
    return gap

def compute_confidence_gap(model, model_path, mem_set, nonmem_set, device):
    print("compute_confidence_gap")
    model = model.to(device)
    print(f"Load model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mem_loader = torch.utils.data.DataLoader(
        mem_set, batch_size=64, shuffle=True, num_workers=8)
    nonmem_loader = torch.utils.data.DataLoader(
        nonmem_set, batch_size=64, shuffle=True, num_workers=8)
    
    total_confidence = []
    with torch.no_grad():
        for inputs, targets in mem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            
            confidence = [
                output[t].item() for output, t in zip(outputs, targets)
            ]
            total_confidence += confidence

    mem_mean_confidence = np.mean(total_confidence)
    
    total_confidence = []
    with torch.no_grad():
        for inputs, targets in nonmem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)

            confidence = [
                output[t].item() for output, t in zip(outputs, targets)
            ]
            total_confidence += confidence

    nonmem_mean_confidence = np.mean(total_confidence)
    
    gap = mem_mean_confidence - nonmem_mean_confidence
    print(f"Confidence gap: member mean conf {mem_mean_confidence:.2f}, non member mean conf {nonmem_mean_confidence:.2f}, gap {gap:.2f}")
    return gap

def compute_loss_gap(model, model_path, mem_set, nonmem_set, device):
    print("compute_loss")
    model = model.to(device)
    print(f"Load model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mem_loader = torch.utils.data.DataLoader(
        mem_set, batch_size=64, shuffle=True, num_workers=8)
    nonmem_loader = torch.utils.data.DataLoader(
        nonmem_set, batch_size=64, shuffle=True, num_workers=8)
    criterion = nn.CrossEntropyLoss()

    losses = []
    with torch.no_grad():
        for inputs, targets in mem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    mem_loss = np.mean(losses)
    
    losses = []
    with torch.no_grad():
        for inputs, targets in nonmem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            
    nonmem_loss = np.mean(losses)
    
    gap = nonmem_loss - mem_loss
    print(f"Loss gap: member loss {mem_loss:.2f}, non member loss {nonmem_loss:.2f}, gap {gap:.2f}")
    return gap

def test_meminf_add_loss(
    save_dir, target_model_dir, shadow_model_dir, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    target_model, shadow_model, args,
):
    # LOG_PATH = PATH + f"_test_meminf_{target_name}/"
    LOG_PATH = save_dir
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    shadow_model_path = osp.join(shadow_model_dir, "shadow.pth")
    target_model_path = osp.join(target_model_dir, "target.pth")

    assert os.path.exists(shadow_model_path)

    last_results, best_results = {}, {}
    
    last_path = osp.join(save_dir, "last.json")
    with open(last_path) as f:
        last_results = json.load(f)
    best_path = osp.join(save_dir, "best.json")
    with open(best_path) as f:
        best_results = json.load(f)
    
    loss_gap = compute_loss_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    best_results['loss_gap'] = loss_gap
    last_results['loss_gap'] = loss_gap
    
    last_path = osp.join(save_dir, "last.json")
    with open(last_path, 'w') as f:
        json.dump(last_results, f, indent=True)
    best_path = osp.join(save_dir, "best.json")
    with open(best_path, 'w') as f:
        json.dump(best_results, f, indent=True)
    
def test_meminf_no_train(
    save_dir, target_model_dir, shadow_model_dir, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    target_model, shadow_model, args,
):
    # LOG_PATH = PATH + f"_test_meminf_{target_name}/"
    LOG_PATH = save_dir
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    shadow_model_path = osp.join(shadow_model_dir, "shadow.pth")
    target_model_path = osp.join(target_model_dir, "target.pth")

    assert os.path.exists(shadow_model_path)

    last_results, best_results = {}, {}
    
    generalization_gap = compute_generalization_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    last_results['generalization_gap'] = generalization_gap
    best_results['generalization_gap'] = generalization_gap
    
    confidence_gap = compute_confidence_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    last_results['confidence_gap'] = confidence_gap
    best_results['confidence_gap'] = confidence_gap
            
    
            
    tag = 'top3'
    print("Prepare top3 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish top3 attack dataset prepare")
    attack_model = PartialAttackModel(3)
    print("top3 attack")
    top3_res_best = attack_top3_no_train(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes, shadow_model_dir
    )
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("top3 attack finished")
    

    # shadow dataset, output
    tag = 'mode0'
    print("Prepare mode0 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish mode0 attack dataset prepare")
    attack_model = PartialAttackModel(num_classes)
    print("Mode0 attack")
    mode0_res_best = attack_mode0_no_train(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes, shadow_model_dir
    )
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("Mode0 attack finished")

    #for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
    # shadow dataset, whitebox
    tag = 'mode3'
    print("Prepare mode3 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish mode3 attack dataset prepare")

    attack_model = WhiteBoxAttackModel(num_classes, total)
    print("Mode3 attack")
    mode3_res_best = attack_mode3_no_train(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes, shadow_model_dir
    )
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("Mode3 attack finished")
    
    
    # #for white box
    # feature_size = get_feature_size(target_model)
    # # shadow dataset, whitebox
    # tag = 'feature'
    # print("Prepare mode whitebox feature attack dataset")
    # attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
    #     target_train, target_test, shadow_train, shadow_test, args.batch_size)
    # print("Finish mode3 whitebox feature attack dataset prepare")
    # attack_model = WhiteBoxFeatureAttackModel(num_classes, feature_size)
    # print("Mode whitebox feature attack")
    # _, feature_res_last, feature_res_best = attack_mode_whitebox_feature(
    #     target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
    #     target_model, shadow_model, attack_model, 1, num_classes)
    # last_results[f"{tag}_last_f1"] = eval(f"{tag}_res_last")[0]
    # last_results[f"{tag}_last_roc_auc"] = eval(f"{tag}_res_last")[1]
    # last_results[f"{tag}_last_acc"] = eval(f"{tag}_res_last")[2]
    # best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    # best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    # best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    # print("Mode whitebox feature attack finished")
    
    last_path = osp.join(save_dir, "last.json")
    with open(last_path, 'w') as f:
        json.dump(last_results, f, indent=True)
    best_path = osp.join(save_dir, "best.json")
    with open(best_path, 'w') as f:
        json.dump(best_results, f, indent=True)


def test_meminf_full(
    save_dir, target_model_dir, shadow_model_dir, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    target_model, shadow_model, args,
):
    # LOG_PATH = PATH + f"_test_meminf_{target_name}/"
    LOG_PATH = save_dir
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    shadow_model_path = osp.join(shadow_model_dir, "shadow.pth")
    target_model_path = osp.join(target_model_dir, "target.pth")
    
    if shadow_model:
        # FILE_PATH = PATH + "_" + "shadow.pth"
        
        if os.path.exists(shadow_model_path):
            print(f"Shadow model exists in {shadow_model_path}")
        else:
            print("Training shadow model")

            shadow_trainloader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
            shadow_testloader = torch.utils.data.DataLoader(
                shadow_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

            loss = nn.CrossEntropyLoss()
            optimizer = optim.SGD(shadow_model.parameters(), lr=args.shadow_lr, momentum=0.9, weight_decay=5e-4)
            shadow_model_logger = Logger(log2file=True, mode="shadow_model", path=LOG_PATH)
            train_shadow_model(
                shadow_model_dir, device, shadow_model, shadow_trainloader, shadow_testloader, 
                args.batch_size, loss, optimizer, shadow_model_logger, args
            )
            print("Shadow model training finished")
            
    last_results, best_results = {}, {}
    
    generalization_gap = compute_generalization_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    last_results['generalization_gap'] = generalization_gap
    best_results['generalization_gap'] = generalization_gap
    # print(generalization_gap)
    
    confidence_gap = compute_confidence_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    last_results['confidence_gap'] = confidence_gap
    best_results['confidence_gap'] = confidence_gap
    # print(confidence_gap)
            
    
            
    tag = 'top3'
    print("Prepare top3 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish top3 attack dataset prepare")
    attack_model = PartialAttackModel(3)
    print("top3 attack")
    _, top3_res_last, top3_res_best = attack_top3(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes
    )
    last_results[f"{tag}_last_f1"] = eval(f"{tag}_res_last")[0]
    last_results[f"{tag}_last_roc_auc"] = eval(f"{tag}_res_last")[1]
    last_results[f"{tag}_last_acc"] = eval(f"{tag}_res_last")[2]
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("top3 attack finished")

    # shadow dataset, output
    tag = 'mode0'
    print("Prepare mode0 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish mode0 attack dataset prepare")
    attack_model = PartialAttackModel(num_classes)
    print("Mode0 attack")
    _, mode0_res_last, mode0_res_best = attack_mode0(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes
    )
    last_results[f"{tag}_last_f1"] = eval(f"{tag}_res_last")[0]
    last_results[f"{tag}_last_roc_auc"] = eval(f"{tag}_res_last")[1]
    last_results[f"{tag}_last_acc"] = eval(f"{tag}_res_last")[2]
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("Mode0 attack finished")

    #for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
    # shadow dataset, whitebox
    tag = 'mode3'
    print("Prepare mode3 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish mode3 attack dataset prepare")

    attack_model = WhiteBoxAttackModel(num_classes, total)
    print("Mode3 attack")
    _, mode3_res_last, mode3_res_best = attack_mode3(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes)
    last_results[f"{tag}_last_f1"] = eval(f"{tag}_res_last")[0]
    last_results[f"{tag}_last_roc_auc"] = eval(f"{tag}_res_last")[1]
    last_results[f"{tag}_last_acc"] = eval(f"{tag}_res_last")[2]
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("Mode3 attack finished")
    
    
    # #for white box
    # feature_size = get_feature_size(target_model)
    # # shadow dataset, whitebox
    # tag = 'feature'
    # print("Prepare mode whitebox feature attack dataset")
    # attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
    #     target_train, target_test, shadow_train, shadow_test, args.batch_size)
    # print("Finish mode3 whitebox feature attack dataset prepare")
    # attack_model = WhiteBoxFeatureAttackModel(num_classes, feature_size)
    # print("Mode whitebox feature attack")
    # _, feature_res_last, feature_res_best = attack_mode_whitebox_feature(
    #     target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
    #     target_model, shadow_model, attack_model, 1, num_classes)
    # last_results[f"{tag}_last_f1"] = eval(f"{tag}_res_last")[0]
    # last_results[f"{tag}_last_roc_auc"] = eval(f"{tag}_res_last")[1]
    # last_results[f"{tag}_last_acc"] = eval(f"{tag}_res_last")[2]
    # best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    # best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    # best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    # print("Mode whitebox feature attack finished")
    
    last_path = osp.join(save_dir, "last.json")
    with open(last_path, 'w') as f:
        json.dump(last_results, f, indent=True)
    best_path = osp.join(save_dir, "best.json")
    with open(best_path, 'w') as f:
        json.dump(best_results, f, indent=True)
    
    
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