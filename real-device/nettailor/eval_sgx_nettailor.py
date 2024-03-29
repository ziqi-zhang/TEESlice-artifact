import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
from pdb import set_trace as st
import argparse

from nettailor.sgx_student_resnet_small_tailor import eval_sgx_nettailor_time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "resnet101"])
    parser.add_argument("--mode", default="GPU", choices=["GPU", "CPU", "Enclave"])
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_repeat", default=5, type=int)
    parser.add_argument("--root", default="nettailor/cifar10val")
    parser.add_argument("--pretrained_dataset", default="cifar100")
    args = parser.parse_args()

    pretrained_subdir = f"{args.pretrained_dataset}-{args.arch}"
    pretrained_path = osp.join(args.root, pretrained_subdir, "checkpoint.pth.tar")
    if not osp.exists(pretrained_path):
        print(f"pretrained_path does not exist")
        print(f"pretrained_path is {pretrained_path}")
        raise RuntimeError

    all_dirs = os.listdir(args.root)
    arch_dirs = [f for f in all_dirs if args.arch in f and ".csv" not in f]
    arch_dirs.remove(pretrained_subdir)

    nettailor_subdirs = [f for f in arch_dirs if "iterative" in f]
    if len(nettailor_subdirs) != 1:
        print("the number of files in nettailor_subdirs is not 1")
        print(f"nettailor_subdirs: {nettailor_subdirs}")
        raise RuntimeError

    nettailor_subdir = nettailor_subdirs[0]
    arch_dirs.remove(nettailor_subdir)
    nettailor_full_subdir = arch_dirs[0]

    nettailor_dir = osp.join(args.root, nettailor_subdir, "checkpoint.pth.tar")
    if not osp.exists(nettailor_dir):
        print(f"nettailor_dir does not exist")
        print(f"nettailor_dir is {nettailor_dir}")
        raise RuntimeError

    nettailor_full_dir = osp.join(args.root, nettailor_full_subdir, "checkpoint.pth.tar")
    if not osp.exists(nettailor_full_dir):
        print(f"nettailor_full_dir does not exist")
        print(f"nettailor_full_dir is {nettailor_full_dir}")
        raise RuntimeError

    inf_times = eval_sgx_nettailor_time(
        arch=args.arch, root=args.root, pretrained_subdir=pretrained_subdir,
        nettailor_subdir=nettailor_subdir, nettailor_full_subdir=nettailor_full_dir,
        batch_size=args.batch_size, num_classes=10, num_repeat=args.num_repeat
    )

    save = pd.DataFrame(inf_times)
    path = osp.join(args.root, f"{args.arch}_{args.batch_size}_{args.num_repeat}.csv")
    save.to_csv(path)


