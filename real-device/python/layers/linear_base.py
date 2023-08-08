import torch

from python.common_torch import mod_move_down
from python.global_config import SecretConfig
from python.linear_shares import LearnableParamTuple, InputGradRemap, WeightGradRemap, set_tensor_name_maybe_quantized
from python.layers.base import SecretLayerBase
from python.timer_utils import NamedTimerInstance, VerboseLevel
from python.torch_utils import compare_expected_actual
from python.enclave_interfaces import GlobalTensor as gt


class SecretLinearLayerBase(SecretLayerBase):
    ForwardOutput = None
    BackwardInput = None
    BackwardWeight = None
    x_shape = None
    w_shape = None
    y_shape = None
    grad_func_for_speed = None

    def __init__(self, sid, LayerName):
        super().__init__(sid, LayerName)

        self.StoreInEnclave = True
        self.SecretOpList = [self.ForwardOutput, self.BackwardInput, self.BackwardWeight, ]
        self.NoRemapTag = self.ForwardOutput.get_tag
        self.IsDummyForS2 = False
        if self.sid == 2:
            return
        self.LearnableParamsList = [("DerWeight", "weight", self.w_shape)]

    def init(self, start_enclave=True):
        super(SecretLinearLayerBase, self).init(start_enclave)
        self.init_secret_op()
        if self.sid != 2:
            self.init_params()

    def init_shape(self):
        # Weight, input, output
        self.ForwardOutput.set_shapes(self.w_shape, self.x_shape, self.y_shape)
        self.BackwardInput.set_shapes(self.w_shape, self.y_shape, self.x_shape)
        self.BackwardWeight.set_shapes(self.y_shape, self.x_shape, self.w_shape)
        self.LearnableParamsList = [LearnableParamTuple(w_name="weight", dw_name="DerWeight", shape=self.w_shape), ]

    def link_tensors(self):
        for k, v in InputGradRemap.items():
            gt.link_tags(self.BackwardInput.get_tag(k, remap=False), self.ForwardOutput.get_tag(v, remap=False))
        for k, v in WeightGradRemap.items():
            gt.link_tags(self.BackwardWeight.get_tag(k, remap=False), self.ForwardOutput.get_tag(v, remap=False))
        # TODO: Reduce reload
        layer_to_op_mapping = [("input", self.ForwardOutput, "Bf"), ("weight", self.ForwardOutput, "Af"),
                               ("output", self.ForwardOutput, "Cf"), ("DerOutput", self.BackwardInput, "Bf"),
                               ("DerOutput", self.BackwardWeight, "Af"), ("DerInput", self.BackwardInput, "Cf"),
                               ("DerWeight", self.BackwardWeight, "Cf")]
        for layer_tensor_name, op, op_tensor_name in layer_to_op_mapping:
            def link_in_linear_layer(a, b, c):
                gt.link_tags(self.get_tag(a, remap=False), b.get_tag(c, remap=False))

            link_in_linear_layer(layer_tensor_name, op, op_tensor_name)
            link_in_linear_layer(layer_tensor_name + "Q", op, op_tensor_name[:-1] + "Q")

        super().link_tensors()

    def init_params(self):
        cpu_w = torch.zeros(self.w_shape)
        torch.nn.init.xavier_normal_(cpu_w, 1)
        self.set_tensor_cpu_enclave("weight", cpu_w)

    # __init__() -> set_eid() -> link_tensors() -> init_secret_op() -> init_params()
    def init_secret_op(self):
        for f in self.SecretOpList:
            f.set_eid(self.get_eid())
            f.init(start_enclave=False)

    def get_output_shape(self):
        return self.y_shape

    def load_tensors(self, w, x, dy, for_quantized=True):
        if w is not None:
            if for_quantized:
                self.ForwardOutput.set_tensor_cpu_enclave("AQ", w)
            else:
                self.ForwardOutput.set_tensor_cpu_enclave("Af", w)
        if x is not None:
            if for_quantized:
                self.ForwardOutput.set_tensor_cpu_enclave("BQ", x)
            else:
                self.ForwardOutput.set_tensor_cpu_enclave("Bf", x)
        if dy is not None:
            if for_quantized:
                self.BackwardWeight.set_tensor_cpu_enclave("AQ", dy)
            else:
                self.BackwardWeight.set_tensor_cpu_enclave("Af", dy)

    def inject_params(self, params):
        if self.sid == -2:
            raise ValueError("S2 has no learnable parameters for injection")
        cpu_w = self.get_cpu("weight")
        cpu_w.copy_(params.weight.data)
        self.transfer_cpu_to_enclave("weight")

    def inject_to_plain(self, plain_layer: torch.nn.Module) -> None:
        if self.sid == -2:
            raise ValueError("S2 has no learnable parameters for injection")
        self.make_sure_cpu_is_latest("weight")
        plain_layer.weight.data.copy_(self.get_cpu("weight"))

    def forward(self, need_quantize=True):
        if self.sid != 2:
            self.forward_tensor_transfer()

        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} forward", verbose_level=VerboseLevel.LAYER):
            self.ForwardOutput.compute(need_quantize=need_quantize)

    def backward(self, need_quantize=True):
        if self.sid != 2:
            self.backward_tensor_transfer()

        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} backward", verbose_level=VerboseLevel.LAYER):
            self.BackwardWeight.compute(need_quantize=need_quantize)
            self.BackwardInput.compute(need_quantize=need_quantize)

    def plain_forward(self, quantized_only=False):
        self.PlainFunc = self.ForwardOutput.target_op
        input_name = set_tensor_name_maybe_quantized("input", quantized_only)
        weight_name = set_tensor_name_maybe_quantized("weight", quantized_only)

        self.make_sure_cpu_is_latest(input_name)
        self.make_sure_cpu_is_latest(weight_name)
        # For timing
        self.requires_grad_on_cpu(weight_name)
        self.requires_grad_on_cpu(input_name)
        torch.set_num_threads(1)
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            forward_result_for_speed = self.PlainFunc(self.get_cpu(weight_name), self.get_cpu(input_name))
        torch.set_num_threads(4)
        self.grad_func_for_speed = forward_result_for_speed.grad_fn
        self.PlainForwardResult = self.PlainFunc(self.get_cpu(weight_name).type(torch.double),
                                                 self.get_cpu(input_name).type(torch.double))
        self.GradFunction = self.PlainForwardResult.grad_fn
        if quantized_only:
            self.PlainForwardResult = mod_move_down(self.PlainForwardResult)
        self.PlainForwardResult = self.PlainForwardResult.type(SecretConfig.dtypeForCpuOp)

    def plain_backward(self, quantized_only=False):
        der_output_name = set_tensor_name_maybe_quantized("DerOutput", quantized_only)
        self.make_sure_cpu_is_latest(der_output_name)
        torch.set_num_threads(1)
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainBackward"):
            self.grad_func_for_speed(self.get_cpu(der_output_name))
        torch.set_num_threads(4)
        self.PlainBackwardResult = self.GradFunction(self.get_cpu(der_output_name).type(torch.double))

    # @profile
    def show_plain_error(self, quantized_only=False):
        output_name = set_tensor_name_maybe_quantized("output", quantized_only)
        der_input_name = set_tensor_name_maybe_quantized("DerInput", quantized_only)
        der_weight_name = set_tensor_name_maybe_quantized("DerWeight", quantized_only)

        self.transfer_enclave_to_cpu(output_name)

        err = compare_expected_actual(self.PlainForwardResult, self.get_cpu(output_name), get_relative=True, show_where_err=True)
        print(f"S{self.sid}: {self.LayerName} Forward Error {err}")

        if self.PlainBackwardResult is None:
            return err
        PlainBackwardInputResult = self.PlainBackwardResult[0]
        PlainBackwardWeightResult = self.PlainBackwardResult[1]
        PlainBackwardWeightResult = self.transpose_weight_grad_for_matmul(PlainBackwardWeightResult)
        if quantized_only:
            PlainBackwardInputResult = mod_move_down(PlainBackwardInputResult)
            PlainBackwardWeightResult = mod_move_down(PlainBackwardWeightResult)

        self.transfer_enclave_to_cpu(der_input_name)
        self.transfer_enclave_to_cpu(der_weight_name)
        # To know the err if the input is completely garbage
        err_grad_weight = compare_expected_actual(PlainBackwardWeightResult, self.get_cpu(der_weight_name),
                                                  get_relative=True)
        err_grad_input = compare_expected_actual(PlainBackwardInputResult, self.get_cpu(der_input_name),
                                                 get_relative=True)

        print(f"S{self.sid}: {self.LayerName} "
              f"BackwardInput : {err_grad_input}, BackwardWeight : {err_grad_weight}")

        return err, err_grad_input, err_grad_weight

    def transpose_weight_grad_for_matmul(self, w):
        return w
