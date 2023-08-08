import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
from pdb import set_trace as st
import argparse
import json

from nettailor_224.sgx_student_resnet_nettailor_freivalds import eval_sgx_nettailor_time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="vgg16_bn", choices=["resnet18", "alexnet", "vgg16_bn"])
    parser.add_argument("--dataset", default="CIFAR10")
    args = parser.parse_args()

    root = "nettailor_224/models_nettailor/models"
    save_root = "nettailor_224/models_nettailor"
    dataset_name = args.dataset
    model_name = args.arch

    results = []
    # for model_name in ["alexnet", "resnet18", "vgg16_bn"]:
    #     for dataset_name in ["CIFAR10", "CIFAR100", "STL10", "UTKFaceRace"]:
    # for model_name in ["alexnet",]:
    #     for dataset_name in ["CIFAR100"]:
    if dataset_name in ["CIFAR10", "STL10"]:
        num_classes = 10
    elif dataset_name == "CIFAR100":
        num_classes = 100
    elif dataset_name == "UTKFaceRace":
        num_classes = 4
    else:
        raise NotImplementedError

    pretrained_subdir = osp.join(
        f"{dataset_name}-{model_name}", "resnet18", 
    )
    pretrained_path = osp.join(root, pretrained_subdir, "checkpoint.pth.tar")
    if not osp.exists(pretrained_path):
        print(f"pretrained_path does not exist")
        print(f"pretrained_path is {pretrained_path}")
        raise RuntimeError


    nettailor_subdir = osp.join(
        f"{dataset_name}-{model_name}", "resnet18-iterative-nettailor-3Skip-T10.0-C0.3-Pruned0", 
    )
    nettailor_dir = osp.join(
        root, nettailor_subdir,
    )
    if not osp.exists(nettailor_dir):
        print(f"nettailor_dir does not exist")
        print(f"nettailor_dir is {nettailor_dir}")
        raise RuntimeError
    inf_times, throughputs = eval_sgx_nettailor_time(
        backbone="resnet18", root=root, pretrained_subdir=pretrained_subdir,
        nettailor_subdir=nettailor_subdir, 
        batch_size=64, num_classes=num_classes, num_repeat=6
    )

    result = {
        "name": f"{model_name}_{dataset_name}"
    }
    mean, std = np.mean(inf_times), np.std(inf_times)
    th_mean, th_std = np.mean(throughputs), np.std(throughputs)
    result["mean"] = mean
    result["std"] = std
    result["inf_times"] = inf_times
    result["th_mean"] = th_mean
    result["th_std"] = th_std
    result["throughputs"] = throughputs

    results.append(result)

    path = osp.join(
        save_root, f"inference_{dataset_name}_{model_name}.json"
    )

    with open(path, "w") as f:
        json.dump(result, f, indent=True)

