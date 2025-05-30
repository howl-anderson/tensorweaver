import numpy as np
from tensorweaver.autodiff.operator import Operator


class LayerNorm(Operator):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, input, weight, bias):
        # mean of last dim
        mean = np.mean(input, axis=-1, keepdims=True)
        # variance of last dim
        var = np.var(input, axis=-1, keepdims=True)
        self.var = var
        self.mean = mean
        # normalize
        std = np.sqrt(var + self.eps)
        self.std = std
        normalized_val = (input - mean) / std
        self.normalized_val = normalized_val
        # scale and shift
        output = weight * normalized_val + bias
        return output

    def backward(self, grad_output):
        input, weight, bias = self.input_data
        N = self.normalized_shape

        # Gradient w.r.t. normalized input
        grad_normalized = grad_output * weight

        # Gradient w.r.t. input
        grad_var = -0.5 * np.sum(
            grad_normalized * (input - self.mean) / (self.var + self.eps) ** 1.5,
            axis=-1,
            keepdims=True,
        )
        grad_mean = -np.sum(grad_normalized / self.std, axis=-1, keepdims=True)

        grad_input = grad_normalized / self.std
        grad_input += 2 * grad_var * (input - self.mean) / N
        grad_input += grad_mean / N

        # Gradient w.r.t. weight and bias
        grad_weight = np.sum(grad_output * self.normalized_val, axis=-1, keepdims=True)
        grad_bias = np.sum(grad_output, axis=-1, keepdims=True)

        return grad_input, grad_weight, grad_bias


def layer_norm(input, normalized_shape, weight, bias, eps):
    return LayerNorm(normalized_shape, eps)(input, weight, bias)
