from python.common_torch import calc_conv2d_output_shape
from python.layers.linear_base import SecretLinearLayerBase
# from python.linear_shares import secret_op_class_factory


class SecretConv2dLayer(SecretLinearLayerBase):
    def __init__(self, sid, layer_name, n_output_channel, filter_hw, batch_size=None, n_input_channel=None,
                 img_hw=None):
        self.batch_size = batch_size
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.img_hw = img_hw
        self.filter_hw = filter_hw
        self.padding = 1

        self.ForwardOutput = secret_op_class_factory(sid, "Conv2d")(layer_name + "ForwardOutput")
        self.BackwardInput = secret_op_class_factory(sid, "Conv2dInputGrad")(layer_name + "BackwardInput")
        self.BackwardWeight = secret_op_class_factory(sid, "Conv2dWeightGrad")(layer_name + "BackwardWeight")
        super().__init__(sid, layer_name)

    def init_shape(self):
        if self.batch_size is None and self.PrevLayer is not None:
            self.x_shape = self.PrevLayer.get_output_shape()
            self.batch_size, self.n_input_channel, self.img_hw, _ = self.x_shape
        else:
            self.x_shape = [self.batch_size, self.n_input_channel, self.img_hw, self.img_hw]
        self.w_shape = [self.n_output_channel, self.n_input_channel, self.filter_hw, self.filter_hw]
        self.y_shape = calc_conv2d_output_shape(self.x_shape, self.w_shape, self.padding)
        super().init_shape()


