import os, sys
import os.path as osp
import json
import numpy as np
from pdb import set_trace as st

ALEXNET_TEE_TIME = 152.7595
ALEXNET_TEE_THR = 6.5568
ALEXNET_GPU_TIME = 2.0249
ALEXNET_GPU_THR = 495.2676

RESNET_TEE_TIME = 130.4687
RESNET_TEE_THR = 7.6729
RESNET_GPU_TIME = 3.5749
RESNET_GPU_THR = 280.5550

VGG_TEE_TIME = 644.2232
VGG_TEE_THR = 1.5523
VGG_GPU_TIME = 9.7032
VGG_GPU_THR = 103.1035

def read_throughput(path):
    with open(path, 'r') as f:
        results = json.load(f)
        return results["Throughput"]


print(''.center(12, ' '), 'AlexNet'.center(30, ' '), "ResNet18".center(30, ' '), "VGG16_BN".center(30, ' '))
print('Black-box'.center(12, ' '), f'{ALEXNET_TEE_THR:.2f}'.center(30, ' '), f'{RESNET_TEE_THR:.2f}'.center(30, ' '), f'{VGG_TEE_THR:.2f}'.center(30, ' '))
print('No-Shield'.center(12, ' '), f"{ALEXNET_GPU_THR:.2f}({ALEXNET_GPU_THR/ALEXNET_TEE_THR:.2f}X)".center(30, ' '), f"{RESNET_GPU_THR:.2f}({RESNET_GPU_THR/RESNET_TEE_THR:.2f}X)".center(30, ' '), f"{VGG_GPU_THR:.2f}({VGG_GPU_THR/VGG_TEE_THR:.2f}X)".center(30, ' '))

throughput_speedup = []
for dataset in ["CIFAR10", "CIFAR100", "STL10", "UTKFaceRace"]:
    alexnet = read_throughput(f"realdevice/layer_analysis/{dataset}_alexnet.json")
    resnet = read_throughput(f"realdevice/layer_analysis/{dataset}_resnet18.json")
    vgg = read_throughput(f"realdevice/layer_analysis/{dataset}_vgg16_bn.json")
    
    throughput_speedup.append(alexnet/ALEXNET_TEE_THR)
    throughput_speedup.append(resnet/RESNET_TEE_THR)
    throughput_speedup.append(vgg/VGG_TEE_THR)

    print(
        f"{dataset}".center(12, ' '),
        f"{alexnet:.2f}({alexnet/ALEXNET_TEE_THR:.2f}X)".center(30, ' '),
        f"{resnet:.2f}({resnet/RESNET_TEE_THR:.2f}X)".center(30, ' '),
        f"{vgg:.2f}({vgg/VGG_TEE_THR:.2f}X)".center(30, ' ')
    )
    
print(f"Average {np.mean(throughput_speedup)}")