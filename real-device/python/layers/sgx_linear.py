import numpy as np
import torch
from pdb import set_trace as st

from python.layers.linear_base import SecretLinearLayerBase
from python.linear_shares import secret_op_class_factory
from python.tensor_loader import TensorLoader
from python.timer_utils import NamedTimerInstance, VerboseLevel
from python.torch_utils import compare_expected_actual

class SecretSGXLinearLayer(SecretLinearLayerBase):
    def __init__(self, sid, LayerName, batch_size, n_output_features, n_input_features=None, is_enclave_mode=True, link_prev=True, link_next=True):
        self.ForwardFuncName = "SGXLinear"
        self.BackwardFuncName = "DerSGXLinear"
        self.PlainFunc = torch.nn.Linear
        self.batch_size = batch_size
        self.n_output_features = n_output_features
        self.n_input_features = n_input_features

        super().__init__(sid, LayerName, link_prev, link_next)
        self.is_enclave_mode = is_enclave_mode
        if self.is_enclave_mode:
            self.StoreInEnclave = True
        else:
            self.ForwardFunc = torch.nn.Linear
            self.StoreInEnclave = False

    def init_shape(self):
        if self.n_input_features is None:
            prev_shape = self.PrevLayer.get_output_shape()
            if len(prev_shape) != 2:
                raise ValueError("The layer previous to a matmul layer should be of 2D.")
            self.n_input_features = prev_shape[-1]

        self.x_shape = [self.batch_size, self.n_input_features]
        self.w_shape = [self.n_output_features, self.n_input_features, ]
        self.y_shape = [self.batch_size, self.n_output_features]
        self.bias_shape = [self.batch_size, self.n_output_features,]
        super().init_shape()

    def init(self, start_enclave=True):
        if self.sid == 2:
            return
        TensorLoader.init(self, start_enclave)
        
        if self.is_enclave_mode:
            self.PlainFunc = self.PlainFunc(self.InputShape[1])
            self.get_cpu("weight").data.copy_(self.PlainFunc.weight.data)
            self.get_cpu("bias").data.copy_(self.PlainFunc.bias.data)
            self.transfer_cpu_to_enclave("weight")
            self.transfer_cpu_to_enclave("bias")
            self.sgx_linear_init(
                self.LayerName,
                "input", "output", "weight", "bias",
                "DerInput", "DerOutput", "DerWeight", "DerBias",
                self.BatchSize, self.n_input_features, self.n_output_features)
        else:
            self.ForwardFunc = self.ForwardFunc(self.InputShape[1])
            self.PlainFunc = self.PlainFunc(self.InputShape[1])
            self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
            self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
            self.set_cpu("weight", list(self.ForwardFunc.parameters())[0].data)
            self.set_cpu("bias", list(self.ForwardFunc.parameters())[1].data)

    def inject_params(self, params):
        if self.sid == -2:
            raise ValueError("S2 has no learnable parameters for injection")
        self.get_cpu("weight").copy_(params.weight.data)
        self.get_cpu("bias").copy_(params.bias.data)
        self.transfer_cpu_to_enclave("weight")
        self.transfer_cpu_to_enclave("bias")

    def inject_to_plain(self, plain_layer: torch.nn.Module) -> None:
        if self.sid == -2:
            raise ValueError("S2 has no learnable parameters for injection")
        self.make_sure_cpu_is_latest("weight")
        self.make_sure_cpu_is_latest("bias")
        plain_layer.weight.data.copy_(self.get_cpu("weight"))
        plain_layer.bias.data.copy_(self.get_cpu("bias"))

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
        if self.sid == 2:
            self.tensor_name_list = {}
            return

        if self.is_enclave_mode:
            NeededTensorNames = [
                ("input", self.InputShape, None),
                ("DerInput", self.InputShape, None),
                ("output", self.OutputShape, None),
                ("DerOutput", self.OutputShape, None),
                ("weight", self.WeightShape, None),
                ("DerWeight", self.WeightShape, None),
                ("bias", self.WeightShape, None),
                ("DerBias", self.WeightShape, None),
            ]
        else:
            NeededTensorNames = [
                ("output", self.OutputShape, None),
                ("DerInput", self.InputShape, None),
                ("input", self.InputShape, None),
                ("weight", self.WeightShape, None),
                ("DerWeight", self.WeightShape, None),
                ("bias", self.WeightShape, None),
                ("DerBias", self.WeightShape, None),
                ("DerOutput", self.OutputShape, None)
            ]

        self.tensor_name_list = NeededTensorNames

    def transpose_weight_grad_for_matmul(self, w):
        return w.t()

    def forward(self):
        if self.sid == 2:
            return
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            if self.is_enclave_mode:
                self.forward_tensor_transfer()
                self.batchnorm_forward(self.LayerName, int(True))
            else:
                self.forward_tensor_transfer()
                self.requires_grad_on_cpu("input")
                self.ForwardFunc.bias.data.copy_(self.get_cpu("bias"))
                self.ForwardFunc.weight.data.copy_(self.get_cpu("weight"))
                self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))

    def plain_forward(self, NeedBackward=False):
        if self.sid == 2:
            return
        if self.is_enclave_mode:
            self.make_sure_cpu_is_latest("input")
            self.make_sure_cpu_is_latest("bias")
            self.make_sure_cpu_is_latest("weight")
            self.requires_grad_on_cpu("input")
            self.PlainFunc.bias.data.copy_(self.get_cpu("bias"))
            self.PlainFunc.weight.data.copy_(self.get_cpu("weight"))
        else:
            self.make_sure_cpu_is_latest("input")
            self.requires_grad_on_cpu("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            torch.set_num_threads(1)
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
            torch.set_num_threads(4)

    def show_plain_error_forward(self):
        if self.sid == 2:
            return
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu("output"), get_relative=True)
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")

