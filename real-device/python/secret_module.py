#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import hashlib

import numpy as np
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from python.timer_utils import NamedTimer

np.random.seed(123)
# For testing
minibatch, inChan, outChan, imgHw, filHw = 512, 256, 128, 8, 3
xShape = [minibatch, inChan, imgHw, imgHw]
wShape = [outChan, inChan, filHw, filHw]

class SecretConfig(object):
    PrimeLimit     = (1 << 14) - 3
    dtypeForCpuMod = torch.int32
    dtypeForCudaMm = torch.double
    dtypeForCpuOp  = torch.float
    dtypeForSave   = torch.float
    worldSize      = 3

def GenRandomUniform(upperBound, size):
    return torch.from_numpy(np.random.uniform(0, upperBound - 1, size=size))

def ModOnCpu(x):
    return x.fmod_(SecretConfig.PrimeLimit)
    #.add_(SecretConfig.PrimeLimit).fmod_(SecretConfig.PrimeLimit)
    # return x.type(torch.double).fmod_(PRIME_LIMIT).add_(PRIME_LIMIT).fmod_(PRIME_LIMIT).type(torch.int16)
    return torch.from_numpy(np.fmod(x.numpy().astype(np.uint16) + PRIME_LIMIT, PRIME_LIMIT).astype(np.int16))

def ModOnGpu(x):
    x.fmod_(SecretConfig.PrimeLimit).add_(SecretConfig.PrimeLimit).fmod_(SecretConfig.PrimeLimit)

AQ = GenRandomUniform(SecretConfig.PrimeLimit, wShape).type(SecretConfig.dtypeForCpuMod)
BQ = GenRandomUniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuMod)
U  = GenRandomUniform(SecretConfig.PrimeLimit, AQ.size()).type(SecretConfig.dtypeForCpuMod)
V  = GenRandomUniform(SecretConfig.PrimeLimit, BQ.size()).type(SecretConfig.dtypeForCpuMod)
A0 = GenRandomUniform(SecretConfig.PrimeLimit, AQ.size()).type(SecretConfig.dtypeForCpuMod)
B0 = GenRandomUniform(SecretConfig.PrimeLimit, BQ.size()).type(SecretConfig.dtypeForCpuMod)

# s consume the dummy self
ConvOp = lambda w, x: torch.conv2d(x, w, padding=1)
MatmOp = lambda w, x: torch.mm(w, x)

TargetOp = ConvOp
idealC = ModOnCpu(TargetOp(AQ.type(torch.double), BQ.type(torch.double))).type(SecretConfig.dtypeForCpuOp)

def initCommunicate(rank, MasterAddr, MasterPort, backend='gloo'):
    os.environ['MASTER_ADDR'] = MasterAddr
    os.environ['MASTER_PORT'] = MasterPort
    dist.init_process_group(backend, rank=rank, world_size=SecretConfig.worldSize)

def WarmingUpCuda():
    NamedTimer.start_timer("WarmingCuda")
    dummyA = GenRandomUniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForSave)
    dummyB = GenRandomUniform(SecretConfig.PrimeLimit, wShape).type(SecretConfig.dtypeForSave)
    dummyC = torch.conv2d(dummyA.cuda().type(SecretConfig.dtypeForCudaMm), dummyB.cuda().type(SecretConfig.dtypeForCudaMm), padding=1)
    NamedTimer.end_timer("WarmingCuda")

def GetTensorError(x, y):
    return np.sum(np.abs((x - y).numpy())) / np.prod(x.size())

class SecretLoader(object):
    def __init__(self):
        pass

    def GetA0(self):
        self.A0 = A0

    def GetB0(self):
        self.B0 = B0

    def GetA1(self):
        self.A1 = ModOnCpu(AQ - A0)

    def GetB1(self):
        self.B1 = ModOnCpu(BQ - B0)

    def GetU(self):
        self.U = U

    def GetV(self):
        self.V = V

    def GetE(self):
        self.E = ModOnCpu(AQ - U)

    def GetF(self):
        self.F = ModOnCpu(BQ - V)

class SecretConv2d(object):
    TargetOp = lambda s, a, b: ConvOp(a, b)
    def __init__(self):
        pass

class SecretMm(object):
    TargetOp = lambda s, a, b: MatmOp(a, b)
    def __init__(self):
        pass

class SecretOpBase(object):
    def __init__(self, name):
        self.OutputShape = 0
        self.name = name

    def GetTensorTag(self, TensorName):
        # print("Generating:", self.name + TensorName, hash(self.name + TensorName) % ((1 << 30) - 1))
        return int(int(hashlib.sha224((self.name + TensorName).encode('utf-8')).hexdigest(), 16) % ((1 << 30) - 1))

    def GetOutputShape(self):
        if self.OutputShape == 0:
            self.OutputShape = list(idealC.size())
            # self.OutputShape = list(\
            #         self.TargetOp(\
            #         AQ.cuda().type(SecretConfig.dtypeForGpuMm),\
            #         AQ.cdua().type(SecretConfig.dtypeForGpuMm))\
            #         .type(SecretConfig.dtypeForSave)\
            #         .size())
        return self.OutputShape

# S0
class SecretBaseS0(SecretOpBase):
    def __init__(self, name):
        super().__init__(name)

    def SecretSharingCompute(self):
        self.C1 = torch.zeros(self.GetOutputShape())
        self.Z  = torch.zeros(self.GetOutputShape())
        flagZ   = torch.zeros(1)

        print("Waiting for start")
        # flagStart = torch.zeros(1)
        # start     = dist.irecv(tensor=flagStart, src=2, tag=self.GetTensorTag("startS0"))
        # start.wait()
        dist.barrier()

        NamedTimer.start("S0: SecretSharingCompute")
        NamedTimer.start("S0: PrepareComm")
        signZ   = dist.isend(tensor=flagZ, dst=2, tag=self.GetTensorTag("signZfromS0"))
        recvC1  = dist.irecv(tensor=self.C1, src=1, tag=self.GetTensorTag("C1"))
        recvZ   = dist.irecv(tensor=self.Z, src=2, tag=self.GetTensorTag("ZtoS0"))
        NamedTimer.end("S0: PrepareComm")

        NamedTimer.start("S0: GetRandom")
        self.GetA0()
        self.GetB0()
        self.GetE()
        self.GetF()
        NamedTimer.end("S0: GetRandom")

        NamedTimer.start("S0: Gpu")
        self.gpuA0 = self.A0.cuda().type(SecretConfig.dtypeForCudaMm)
        self.gpuB0 = self.B0.cuda().type(SecretConfig.dtypeForCudaMm)
        self.gpuE  = self.E.cuda().type(SecretConfig.dtypeForCudaMm)
        self.gpuF  = self.F.cuda().type(SecretConfig.dtypeForCudaMm)

        # Gpu Phase
        self.gpuC0 = self.TargetOp(self.gpuA0, self.gpuF)\
                + self.TargetOp(self.gpuE, self.gpuB0)\
                - self.TargetOp(self.gpuE, self.gpuF)
        ModOnGpu(self.gpuC0)

        self.gpuC0 = self.gpuC0.type(SecretConfig.dtypeForSave)
        self.C0    = self.gpuC0.cpu().type(SecretConfig.dtypeForCpuOp)
        NamedTimer.end("S0: Gpu")
        
        NamedTimer.start("S0: Finalnetwork")
        NamedTimer.end("S0: Finalnetwork")

        NamedTimer.start("S0: Recon")
        sendC0 = dist.isend(tensor=self.C0, dst=1, tag=self.GetTensorTag("C0"))
        recvZ.wait()
        NamedTimer.end("S0: Recon", "recvZ.wait()")
        self.C = self.C0 + self.Z
        NamedTimer.end("S0: Recon", "self.C = self.C0 + self.Z")
        recvC1.wait()
        NamedTimer.end("S0: Recon", "recvC1.wait()")
        self.C += self.C1
        NamedTimer.end("S0: Recon", "self.C += self.C1")
        ModOnCpu(self.C)
        NamedTimer.end("S0: Recon")

        NamedTimer.end("S0: SecretSharingCompute")
        # signZ.wait()
        # sendC0.wait()

        # print("S0: C0", self.C0[0, 0, 0, 0])
        # print("S0: C1", self.C1[0, 0, 0, 0])
        # print("S0: Z ", self.Z [0, 0, 0, 0])
        # print("S0: idealC", idealC[0, 0, 0, 0])

        print("Recon Err:", GetTensorError(self.C, idealC))

class SecretConv2dS0(SecretBaseS0, SecretConv2d, SecretLoader):
    def __init__(self, name):
        super().__init__(name)

class SecretMatmulS0(SecretBaseS0, SecretMm, SecretLoader):
    def __init__(self, name):
        super().__init__(name)

# S1
class SecretBaseS1(SecretOpBase):
    def __init__(self, name):
        super().__init__(name)

    def SecretSharingCompute(self):
        self.C  = torch.zeros(self.GetOutputShape())
        self.C0 = torch.zeros(self.GetOutputShape())
        self.Z  = torch.zeros(self.GetOutputShape())
        flagZ   = torch.zeros(1)

        print("Waiting for start")
        # flagStart = torch.zeros(1)
        # start     = dist.irecv(tensor=flagStart, src=2, tag=self.GetTensorTag("startS1"))
        # start.wait()
        dist.barrier()

        NamedTimer.start("S1: SecretSharingCompute")
        signZ   = dist.isend(tensor=flagZ, dst=2, tag=self.GetTensorTag("signZfromS1"))
        recvC0  = dist.irecv(tensor=self.C0, src=0, tag=self.GetTensorTag("C0"))
        recvZ   = dist.irecv(tensor=self.Z, src=2, tag=self.GetTensorTag("ZtoS1"))

        NamedTimer.start("S1: GetRandom")
        self.GetA1()
        self.GetB1()
        self.GetE()
        self.GetF()
        NamedTimer.end("S1: GetRandom")

        NamedTimer.start("S1: Gpu")
        self.gpuA1 = self.A1.cuda(non_blocking=True).type(SecretConfig.dtypeForCudaMm)
        self.gpuB1 = self.B1.cuda(non_blocking=True).type(SecretConfig.dtypeForCudaMm)
        self.gpuE  = self.E.cuda(non_blocking=True).type(SecretConfig.dtypeForCudaMm)
        self.gpuF  = self.F.cuda(non_blocking=True).type(SecretConfig.dtypeForCudaMm)

        # Gpu Phase
        self.gpuC1 = self.TargetOp(self.gpuA1, self.gpuF) + self.TargetOp(self.gpuE, self.gpuB1)
        ModOnGpu(self.gpuC1)

        self.gpuC1 = self.gpuC1.type(SecretConfig.dtypeForSave)
        self.C1    = self.gpuC1.cpu().type(SecretConfig.dtypeForCpuOp)
        NamedTimer.end("S1: Gpu")

        # NamedTimer.start("S1: Finalnetwork")
        # NamedTimer.end("S1: Finalnetwork")
        NamedTimer.start("S1: Recon")
        sendC1 = dist.isend(tensor=self.C1, dst=0, tag=self.GetTensorTag("C1"))
        recvZ.wait()
        NamedTimer.end("S1: Recon", "recvZ.wait()")
        self.C = self.C1 + self.Z
        NamedTimer.end("S1: Recon", "self.C = self.C1 + self.Z")
        recvC0.wait()
        NamedTimer.end("S1: Recon", "recvC0.wait()")
        self.C += self.C0
        NamedTimer.end("S1: Recon", "self.C += self.C0")
        ModOnCpu(self.C)
        NamedTimer.end("S1: Recon")

        # print("S1: C0", self.C0[0, 0, 0, 0])
        # print("S1: C1", self.C1[0, 0, 0, 0])
        # print("S1: Z ", self.Z [0, 0, 0, 0])
        # print("S1: C ", self.C [0, 0, 0, 0])

        NamedTimer.end("S1: SecretSharingCompute")

        # signZ.wait()
        # sendC1.wait()

        print("S1: Recon Err:", GetTensorError(self.C, idealC))

class SecretConv2dS1(SecretBaseS1, SecretConv2d, SecretLoader):
    def __init__(self, name):
        super().__init__(name)

class SecretMatmulS1(SecretBaseS1, SecretMm, SecretLoader):
    def __init__(self, name):
        super().__init__(name)

# S2
class SecretBaseS2(SecretOpBase):
    def __init__(self, name):
        super().__init__(name)

    def SecretSharingCompute(self):

        print("Waiting for start")
        # flagStart = torch.zeros(1)
        # startS0   = dist.isend(tensor=flagStart, dst=0, tag=self.GetTensorTag("startS0"))
        # startS1   = dist.isend(tensor=flagStart, dst=1, tag=self.GetTensorTag("startS1"))
        # startS0.wait()
        # startS1.wait()
        dist.barrier()

        NamedTimer.start("S2: SecretSharingCompute")

        NamedTimer.start("S2: Gpu")
        self.GetU()
        self.GetV()

        self.gpuU = self.U.cuda().type(SecretConfig.dtypeForCudaMm)
        self.gpuV = self.V.cuda().type(SecretConfig.dtypeForCudaMm)

        self.gpuZ = self.TargetOp(self.gpuU, self.gpuV)
        ModOnGpu(self.gpuZ)

        self.Z = self.gpuZ.cpu().type(SecretConfig.dtypeForCpuOp)
        NamedTimer.end("S2: Gpu")

        NamedTimer.start("S2: Finalnetwork")
        sendZToS0 = dist.isend(tensor=self.Z, dst=0, tag=self.GetTensorTag("ZtoS0"))
        sendZToS1 = dist.isend(tensor=self.Z, dst=1, tag=self.GetTensorTag("ZtoS1"))

        sendZToS0.wait()
        sendZToS1.wait()
        NamedTimer.end("S2: Finalnetwork")

        NamedTimer.end("S2: SecretSharingCompute")

class SecretConv2dS2(SecretBaseS2, SecretConv2d, SecretLoader):
    def __init__(self, name):
        super().__init__(name)

class SecretMatmulS2(SecretBaseS2, SecretMm, SecretLoader):
    def __init__(self, name):
        super().__init__(name)

def initProcesses(sid, MasterAddr, MasterPort):
    initCommunicate(sid, MasterAddr, MasterPort)
    LayerName = "SingleLayer"

    if sid   == 0: proc = SecretConv2dS0(LayerName)
    elif sid == 1: proc = SecretConv2dS1(LayerName)
    elif sid == 2: proc = SecretConv2dS2(LayerName)

    WarmingUpCuda()
    proc.SecretSharingCompute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", "-s", 
            type=int,
            default=-1,
            help="The ID of the server")
    parser.add_argument("--ip", 
            dest="MasterAddr",
            default="127.0.0.1",
            help="The Master Address for communication")
    parser.add_argument("--port", 
            dest="MasterPort",
            default="29501",
            help="The Master Port for communication")
    args = parser.parse_args()
    sid = args.sid
    MasterAddr = args.MasterAddr
    MasterPort = args.MasterPort

    if sid == -1:
        processes = []
        for rank in range(SecretConfig.worldSize):
            p = Process(target=initProcesses, args=(rank, MasterAddr, MasterPort))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        initProcesses(sid, MasterAddr, MasterPort)
