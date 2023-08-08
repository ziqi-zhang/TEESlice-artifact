import os, sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models

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


def train_model(PATH, device, train_set, test_set, model, use_DP=False, noise=None, norm=None):
    epochs = 100
    debug = gol.get_value("debug")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=PATH)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    
    model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, logger, epochs)
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, )
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target training")

        acc_train = model.train()
        logger.add_line("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    FILE_PATH = PATH + "_target.pth"
    model.saveModel(FILE_PATH)
    logger.add_line(f"Saved target model to {FILE_PATH}!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting

def train_eval_model(PATH, device, train_set, test_set, model, use_DP, noise, norm):
    
    epochs = 100
    debug = gol.get_value("debug")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=PATH)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    
    model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, logger,epochs)
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, )
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target training")

        acc_train = model.train()
        logger.add_line("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    FILE_PATH = PATH + "_eval.pth"
    model.saveModel(FILE_PATH)
    logger.add_line(f"Saved target model to {FILE_PATH}!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting


def test_meminf(
    PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    target_model, shadow_model, target_name = "target"
):
    LOG_PATH = PATH + f"_test_meminf_{target_name}/"
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    batch_size = 64
    if shadow_model:
        FILE_PATH = PATH + "_" + "shadow.pth"
        if os.path.exists(FILE_PATH):
            print(f"Shadow model exists in {FILE_PATH}")
        else:
            print("Training shadow model")

            shadow_trainloader = torch.utils.data.DataLoader(
                shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
            shadow_testloader = torch.utils.data.DataLoader(
                shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)

            loss = nn.CrossEntropyLoss()
            optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
            shadow_model_logger = Logger(log2file=True, mode="shadow_model", path=LOG_PATH)
            train_shadow_model(
                PATH+"_", device, shadow_model, shadow_trainloader, shadow_testloader, 
                0, 0, 0, batch_size, loss, optimizer, shadow_model_logger
            )
            print("Shadow model training finished")
            
    print("Prepare top3 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)
    print("Finish top3 attack dataset prepare")
    attack_model = PartialAttackModel(3)
    print("top3 attack")
    attack_top3(
        PATH + f"_{target_name}.pth", PATH + "_" + "shadow.pth", LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes
    )
    print("top3 attack finished")

        
    print("Prepare mode0 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)
    print("Finish mode0 attack dataset prepare")
    attack_model = PartialAttackModel(num_classes)
    print("Mode0 attack")
    attack_mode0(
        PATH + f"_{target_name}.pth", PATH + "_" + "shadow.pth", LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes
    )
    print("Mode0 attack finished")

    # print("Prepare mode1 attack dataset")
    # attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)
    # print("Finish mode1 attack dataset prepare")
    # attack_model = PartialAttackModel(num_classes)
    # print("Mode1 attack")
    # attack_mode1(
    #     PATH + f"_{target_name}.pth", LOG_PATH, device, attack_trainloader, attack_testloader, 
    #     target_model, attack_model, 1, num_classes
    # )
    # print("Mode1 attack finished")
    
    #for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
    
    # print("Prepare mode2 attack dataset")
    # attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(
    #     target_train, target_test, batch_size)
    # print("Finish mode2 attack dataset prepare")
    # attack_model = WhiteBoxAttackModel(num_classes, total)
    # print("Mode2 attack")
    # attack_mode2(
    #     PATH + f"_{target_name}.pth", LOG_PATH, device, attack_trainloader, attack_testloader, 
    #     target_model, attack_model, 1, num_classes)
    # print("Mode2 attack finished")
    
    print("Prepare mode3 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)
    print("Finish mode3 attack dataset prepare")
    attack_model = WhiteBoxAttackModel(num_classes, total)
    print("Mode3 attack")
    attack_mode3(
        PATH + f"_{target_name}.pth", PATH + "_" + "shadow.pth", LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes)
    print("Mode3 attack finished")
    



def test_modsteal(PATH, device, train_set, test_set, target_model, attack_model):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)

    loss = nn.MSELoss()
    optimizer = optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9)

    attacking = train_steal_model(
        train_loader, test_loader, target_model, attack_model, PATH + "_target.pth", PATH + "_modsteal.pth", device, 64, loss, optimizer)

    for i in range(100):
        print("[Epoch %d/%d] attack training"%((i+1), 100))
        attacking.train_with_same_distribution()
    
    print("Finished training!!!")
    attacking.saveModel()
    acc_test, agreement_test = attacking.test()
    print("Saved Target Model!!!\nstolen test acc = %.3f, stolen test agreement = %.3f\n"%(acc_test, agreement_test))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")



    TARGET_DIR = target_root + name + "/"
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    TARGET_PATH = TARGET_DIR + name    

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(name, attr, root, "resnet18", pretrained=pretrained)
    train_model(TARGET_PATH, device, target_train, target_test, target_model)
    test_meminf(
        TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
        target_model, shadow_model, "target"
    )


    name = "cifar100"
    attr = "attr"
    
    TARGET_DIR = target_root + name + "/"
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    TARGET_PATH = TARGET_DIR + name    

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(name, attr, root, "resnet18", pretrained=pretrained)
    train_model(TARGET_PATH, device, target_train, target_test, target_model)
    test_meminf(
        TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, 
        target_model, shadow_model, "target"
    )