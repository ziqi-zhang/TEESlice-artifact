from python.layers.linear_base import SecretLinearLayerBase
# from python.linear_shares import secret_op_class_factory


class SecretMatmulLayer(SecretLinearLayerBase):
    def __init__(self, sid, LayerName, batch_size, n_output_features, n_input_features=None):
        self.batch_size = batch_size
        self.n_output_features = n_output_features
        self.n_input_features = n_input_features

        self.ForwardOutput = secret_op_class_factory(sid, "Matmul")(LayerName + "ForwardOutput")
        self.BackwardInput = secret_op_class_factory(sid, "MatmulInputGrad")(LayerName + "BackwardInput")
        self.BackwardWeight = secret_op_class_factory(sid, "MatmulWeightGrad")(LayerName + "BackwardWeight")
        super().__init__(sid, LayerName)

    def init_shape(self):
        if self.n_input_features is None:
            prev_shape = self.PrevLayer.get_output_shape()
            if len(prev_shape) != 2:
                raise ValueError("The layer previous to a matmul layer should be of 2D.")
            self.n_input_features = prev_shape[-1]

        self.x_shape = [self.batch_size, self.n_input_features]
        self.w_shape = [self.n_output_features, self.n_input_features, ]
        self.y_shape = [self.batch_size, self.n_output_features]
        super().init_shape()


    def transpose_weight_grad_for_matmul(self, w):
        return w.t()
