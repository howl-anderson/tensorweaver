import numpy as np
from tensorweaver.layers.layer import Layer
from tensorweaver.operators.layer_norm import layer_norm
from tensorweaver.parameter import Parameter


class LayerNorm(Layer):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = Parameter(np.random.uniform(0, 1, normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))

    def forward(self, input):
        return layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
