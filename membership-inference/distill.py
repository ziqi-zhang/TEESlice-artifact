import os, sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from doctor.meminf import *
from doctor.modinv import *
from doctor.attrinf import *
from doctor.modsteal import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *
# from log_utils import *
# import vgg32
import gol


def distill_model(PATH, device, train_set, test_set, model, use_DP, noise, norm, num_classes):
    epochs = 50

        
    debug = gol.get_value("debug")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=PATH)
    logger.add_line(str(train_set))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=0 if debug else 8)
    
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, logger)
    student = vgg32.vgg11(pretrained=False, num_classes=num_classes)
    teacher = copy.deepcopy(model)
    teacher_path = PATH + "_target.pth"
    model = distillation_training(teacher_path, train_loader, test_loader, student, teacher, device, logger, epochs)
    logger.add_line(f"Total epochs {epochs}")
    
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target distill training")

        acc_train = model.train()
        logger.add_line("target distill testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    FILE_PATH = PATH + "_distill_target.pth"
    model.saveModel(FILE_PATH)
    logger.add_line("Saved target distill model!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting

def identify_outlier(teacher, dataset):
    debug = gol.get_value("debug")
    data_train_loader_noshuffle = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)
    celoss = torch.nn.CrossEntropyLoss(reduction = 'none').cuda()

    value = []
    pred_list = []
    index = 0
    print("Identifying outlier")
    teacher.eval()
    with torch.no_grad():
        for i,(inputs, labels) in enumerate(data_train_loader_noshuffle):
            inputs = inputs.cuda()
            outputs = teacher(inputs)
            pred = outputs.data.max(1)[1]
            loss = celoss(outputs, pred)
            value.append(loss.detach().clone())
            index += inputs.shape[0]
            pred_list.append(pred.detach())
            if debug:
                break
    print("Finish identifying outlier")
    return torch.cat(value,dim=0), torch.cat(pred_list,dim=0) 
    
def select_distill_model(
    PATH, device, train_set, test_set, model, use_DP, noise, norm, name, mode="label"
):
    debug = gol.get_value("debug")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name + f"-{mode}", path=PATH)
    epochs = 50
    
    teacher = copy.deepcopy(model)
    teacher_path = PATH + "_target.pth"
    teacher.load_state_dict(torch.load(teacher_path))
    teacher = teacher.to(device)

    loss_thred = 0.1
    if name in ["UTKFace", "lfw", "small-celeba"]:

        _, cacd_dataset, _, _ = get_model_dataset(proxy_name, attr=attr, root=root)
        # proxy_dataset = torch.utils.data.ConcatDataset([casia_dataset, imdb_dataset, cacd_dataset])
        proxy_dataset = cacd_dataset
        loss_thred = 0.3
    elif name == "stl10":

        _, proxy_dataset, _, _ = get_model_dataset(proxy_name, attr=attr, root=root)
        loss_thred = 0.5
    elif name == "cifar100":

        _, proxy_dataset, _, _ = get_model_dataset(proxy_name, attr=attr, root=root)
        
        loss_thred = 0.7
    else:
        raise NotImplementedError

    logger.add_line(f"Raw proxy dataset length {len(proxy_dataset)}, root {root}")
    print(f"Raw proxy dataset length {len(proxy_dataset)}, root {root}")

    # # num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(name, attr, root)
    # _, proxy_dataset, _, _ = get_model_dataset(name, attr=attr, root=root)
    data_train_select_path = PATH + "_select_dataset.pt"
    if not os.path.exists(data_train_select_path):
    
        value, pred = identify_outlier(teacher, proxy_dataset)
        num_loss_thred = sum(value < loss_thred)
        loss_select_index = value.topk(num_loss_thred, largest=False)[1]
        logger.add_line(str(proxy_dataset))
        logger.add_line(f"Default proxy_dataset {len(proxy_dataset)}, loss_select {len(loss_select_index)}")
        logger.add_line(f"Total epochs {epochs}")

        # plot_value_dist(value, positive_index, pred)
        positive_index = loss_select_index.tolist()
        data_train_select = torch.utils.data.Subset(proxy_dataset, positive_index)
        # torch.save(data_train_select, data_train_select_path)
        if name in ["cifar100", "stl10"]:
            torch.save(positive_index, data_train_select_path)
        else:
            torch.save(data_train_select, data_train_select_path)
    else:
        data_train_select = torch.load(data_train_select_path)
        logger.add_line(f"Load selected dataset len {len(data_train_select)}")
        
        if name in ["cifar100", "stl10"]:
            positive_index = torch.load(data_train_select_path)
            data_train_select = torch.utils.data.Subset(proxy_dataset, positive_index)
        else:
            data_train_select = torch.load(data_train_select_path)
        print(f"Proxy dataset exists, length {len(data_train_select)}")
        logger.add_line(f"proxy dataset length {len(data_train_select)}")


    train_loader = torch.utils.data.DataLoader(
        data_train_select, batch_size=128, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=True, num_workers=0 if debug else 8)
    
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, logger)
    
    if mode == "label":
        model = label_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    elif mode == "feature":
        model = feature_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    elif mode == "shallowfeature":
        model = shallow_feature_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    elif mode == "output":
        model = output_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    else:
        raise NotImplementedError
    
    
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target distill training")

        acc_train = model.train()
        logger.add_line("target distill testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    FILE_PATH = PATH + f"_{mode}_target.pth"
    model.saveModel(FILE_PATH)
    logger.add_line(f"Saved target distill model to {FILE_PATH}!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting


def medmnist_proxy_distill_model(
    PATH, device, train_set, test_set, val_set, model, use_DP, noise, norm, name, mode="label"
):
    debug = gol.get_value("debug")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name + f"-{mode}", path=PATH)
    epochs = 50
    
    teacher = copy.deepcopy(model)
    teacher_path = PATH + "_target.pth"
    teacher.load_state_dict(torch.load(teacher_path))
    teacher = teacher.to(device)

    train_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=True, num_workers=0 if debug else 8)
    
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, logger)
    
    if mode == "label":
        model = distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    elif mode == "feature":
        model = feature_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    elif mode == "shallowfeature":
        model = shallow_feature_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    elif mode == "output":
        model = output_distillation_training(teacher_path, train_loader, test_loader, model, teacher, device, logger, epochs)
    else:
        raise NotImplementedError
    
    
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target distill training")

        acc_train = model.train()
        logger.add_line("target distill testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    FILE_PATH = PATH + f"_{mode}_target.pth"
    model.saveModel(FILE_PATH)
    logger.add_line(f"Saved target distill model to {FILE_PATH}!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting

def medmnist_distill_model(PATH, device, train_set, test_set, target_val, model, use_DP, noise, norm, num_classes):
    epochs = 50

    debug = gol.get_value("debug")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=PATH)
    logger.add_line(str(train_set))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=0 if debug else 8)
    
    # model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, logger)
    student = vgg32.vgg11(pretrained=False, num_classes=num_classes)
    teacher = copy.deepcopy(model)
    teacher_path = PATH + "_target.pth"
    model = distillation_training(teacher_path, train_loader, test_loader, student, teacher, device, logger, epochs)
    logger.add_line(f"Total epochs {epochs}")
    
    acc_train = 0
    acc_test = 0

    for i in range(epochs):
        logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
        logger.add_line("target distill training")

        acc_train = model.train()
        logger.add_line("target distill testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        logger.add_line('The overfitting rate is %s' % overfitting)
        if gol.get_value("debug"):
            break

    FILE_PATH = PATH + "_distill_target.pth"
    model.saveModel(FILE_PATH)
    logger.add_line("Saved target distill model!!!")
    logger.add_line("Finished training!!!")

    return acc_train, acc_test, overfitting
