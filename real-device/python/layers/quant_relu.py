import torch
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.basic_utils import ExecutionModeOptions
from python.torch_utils import compare_expected_actual
from python.timer_utils import NamedTimerInstance, VerboseLevel
from python.quantize_net import my_quantize_tensor, my_dequantize_tensor
from python.enclave_interfaces import EnclaveInterface, GlobalTensor

import ctypes as C
from ctypes.util import find_library
import numpy as np

class SecretEnclaveQuantReLULayer(SecretActivationLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False, merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        self.ForwardFuncName = "ReLU"
        self.BackwardFuncName = "DerReLU"
        self.PlainFunc = torch.nn.ReLU

        self.generate_quant_tensor_name_list()
        
        self.outputQ = None


    def init(self, start_enclave=True):
        super().init(start_enclave)
        self.PlainFunc = self.PlainFunc()
        self.init_enclave_quant_tensors()
        self.init_cpu_quant_tensors()

    def init_shape(self):
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
        assert self.InputShape[1]%4 == 0
        self.QuantizedInputShape = [self.InputShape[0], self.InputShape[1]//4, self.InputShape[2], self.InputShape[3]]

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return
        if len(self.InputShape) == 4:
            # self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/262144+1/2))), 262144, 1, 1]
            self.Shapefortranspose = [int(round(((self.InputShape[0] * self.InputShape[1] * self.InputShape[2] * self.InputShape[3])/602112+1/2))), 602112, 1, 1]
            
        else:
            self.Shapefortranspose = self.InputShape
        NeededTensorNames = [("output", self.OutputShape, None),
                            ("handle", self.HandleShape, None),
                            # ("DerInput", self.InputShape, None),
                            ("input", self.InputShape, None),
                            # ("quant_input", self.QuantizedInputShape, None),
                            # ("quant_output", self.QuantizedInputShape, None),
                            ("inputtrans", self.Shapefortranspose, None),
                            ("outputtrans", self.Shapefortranspose, None),
                            ]
        self.tensor_name_list = NeededTensorNames
    
    def generate_quant_tensor_name_list(self):
        NeededTensorNames = [
                            ("quant_input", self.InputShape, None),
                            ("quant_output", self.InputShape, None),
                            # ("inputtrans", self.Shapefortranspose, None),
                            # ("outputtrans", self.Shapefortranspose, None),
                            ]
        self.quant_tensor_name_list = NeededTensorNames
    
    def init_enclave_quant_tensors(self):
        self.generate_quant_tensor_name_list()
        for TensorName, shape, SeedList in self.quant_tensor_name_list:
            if shape is None:
                raise ValueError("The shape is None. Please setup the shape before init_enclave_tensor")
            self.init_enclave_quant_tensor(TensorName, shape)
            if SeedList is None:
                continue
            for seed in SeedList:
                self.set_seed(TensorName, seed)

    def init_cpu_quant_tensors(self):
        self.generate_quant_tensor_name_list()
        for TensorName, shape, _ in self.quant_tensor_name_list:
            self.generate_cpu_quant_tensor(TensorName, shape)


    def forward(self):
        if (self.PrevLayer is not None and self.PrevLayer.EnclaveMode is not ExecutionModeOptions.Enclave) and (self.NextLayer is not None and self.NextLayer.EnclaveMode is not ExecutionModeOptions.Enclave):
            print("Quant forward")
            self.quant_forward()
        else:
            print("Float forward")
            self.float_forward()

    def float_forward(self):
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            self.ForwardFunc = self.floatrelufunc
            self.BackwardFunc = self.floatrelubackfunc
        elif self.EnclaveMode is ExecutionModeOptions.CPU:
            self.ForwardFunc = torch.nn.ReLU
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            self.ForwardFunc = torch.nn.ReLU

        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} float Forward", verbose_level=VerboseLevel.LAYER):
            with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                self.forward_tensor_transfer()
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} ForwardFunc", verbose_level=VerboseLevel.LAYER):
                    self.ForwardFunc("input", "output")
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.CPU and torch.sum(self.get_cpu("input").abs()) == 0:
                    raise RuntimeError(f"{self.LayerName}: SGX input not load")
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                if self.PrevLayer.EnclaveMode is not ExecutionModeOptions.GPU and torch.sum(self.get_gpu("input").abs()) == 0:
                    raise RuntimeError(f"{self.LayerName}: SGX input not load")
                self.set_gpu("output", self.ForwardFunc(self.get_gpu("input")))
            else:
                raise RuntimeError


    def quant_forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} QuantReLULayerUnitForward", verbose_level=VerboseLevel.LAYER):
            with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} QuantReLULayerUnitInputPreprocess", verbose_level=VerboseLevel.LAYER):
                if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.CPU:
                    input = self.get_cpu("input")
                elif self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.GPU:
                    input = self.get_gpu("input")
                else:
                    
                    raise RuntimeError(self.LayerName, " Prev layer not in CPU nor GPU")
                
                inputQ, exp, bits = my_quantize_tensor(input)
                # print("Python quant relu exp ", exp, " bits ", bits)
                # print(inputQ)

                # print("Before------------ ", GlobalTensor.cpu_tensor.keys())
                if self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.CPU:
                    self.set_cpu("quant_input", inputQ)
                    self.quant_transfer_cpu_to_enclave("quant_input")
                elif self.PrevLayer is not None and self.PrevLayer.EnclaveMode is ExecutionModeOptions.GPU:
                    self.set_gpu("quant_input", inputQ)
                    self.quant_transfer_gpu_to_cpu("quant_input")
                    self.quant_transfer_cpu_to_enclave("quant_input")
                else:
                    raise RuntimeError("Prev layer not in CPU nor GPU")
                # print("After------------ ", GlobalTensor.cpu_tensor.keys())
                # self.forward_quant_tensor_transfer(transfer_tensor = "quant_input")

            with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} QuantReLULayerEnclaveForward", verbose_level=VerboseLevel.LAYER):
                self.quant_relunew("quant_input", "quant_output", self.InputShape, exp, bits)
                # self.get_quant_tensor("quant_output", inputQ)
                # self.set_cpu("quant_output", inputQ)

            with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} QuantReLULayerUnitOutputPreprocess", verbose_level=VerboseLevel.LAYER):

                self.quant_transfer_enclave_to_cpu("quant_output")
                outputQ = self.get_cpu("quant_output")

                if self.NextLayer is not None and self.NextLayer.EnclaveMode is ExecutionModeOptions.CPU:
                    # input = self.get_cpu("input")
                    output = my_dequantize_tensor(outputQ, exp, bits)
                    self.set_cpu("output", output)
                elif self.NextLayer is not None and self.NextLayer.EnclaveMode is ExecutionModeOptions.GPU:
                    outputQ = outputQ.cuda()
                    output = my_dequantize_tensor(outputQ, exp, bits)
                    self.set_gpu("output", output)
                else:
                    raise RuntimeError("Prev layer not in CPU nor GPU")
                
                # self.quant_transfer_enclave_to_cpu("quant_output")
                # self.outputQ = self.get_cpu("quant_output")
                
                # self.set_cpu("output", output)

    def floatrelufunc(self, namein, nameout):
        return self.relunew(namein, nameout, self.InputShape)

    def floatrelubackfunc(self, nameout, namedout, namedin):
        return self.relubackward(nameout, namedout, namedin, self.InputShape)

    def get_tensor(self, name, tensor):
        tensor.data.copy_( self.get_cpu(name).data )
    # def quant_relufunc(self, namein, nameout):
    #     return self.quant_relunew(namein, nameout, self.QuantizedInputShape)

    def relubackfunc(self, nameout, namedout, namedin):
        return self.relubackward(nameout, namedout, namedin, self.InputShape)

    def show_plain_error_forward(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=False, show_values=False)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

