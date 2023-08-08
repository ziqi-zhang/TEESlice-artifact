import os, sys
import os.path as osp
import json
from pdb import set_trace as st
import numpy as np

EnclaveConvForwardPercent, GPUConvForwardPercent, DataTransferPercent, NonLinearForwardPercent = [], [], [], []

for dataset in ["CIFAR10", "CIFAR100", "STL10", "UTKFaceRace"]:
    for model in ["alexnet", "resnet18", "vgg16_bn"]:
        with open(f"realdevice/layer_analysis_bs4/{dataset}_{model}.json", 'r') as f:
            results = json.load(f)
            EnclaveConvForwardPercent.append(results["EnclaveConvForwardPercent"])
            GPUConvForwardPercent.append(results["GPUConvForwardPercent"])
            DataTransferPercent.append(results["DataTransferPercent"])
            NonLinearForwardPercent.append(results["NonLinearForwardPercent"])
            
            # print(f"{dataset}_{model}".center(20, ' '), f"{results['Throughput']:.2f}".center(20, ' '), f"{results['Time']:.2f}".center(20, ' '))
            
EnclaveConvForwardPercent = np.mean(EnclaveConvForwardPercent)
GPUConvForwardPercent = np.mean(GPUConvForwardPercent)
DataTransferPercent = np.mean(DataTransferPercent)
NonLinearForwardPercent = np.mean(NonLinearForwardPercent)

print("EnclaveConvForwardPercent: ".ljust(30," "), f"{EnclaveConvForwardPercent:.2f}%")
print("GPUConvForwardPercent: ".ljust(30," "), f"{GPUConvForwardPercent:.2f}%")
print("DataTransferPercent: ".ljust(30," "), f"{DataTransferPercent:.2f}%")
print("NonLinearForwardPercent: ".ljust(30," "), f"{NonLinearForwardPercent:.2f}%")
