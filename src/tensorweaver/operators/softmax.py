import numpy as np
from tensorweaver.autodiff.operator import Operator


class Softmax(Operator):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.output = None

    def forward(self, x):
        # Compute softmax with numerical stability
        x_max = np.max(x, axis=self.dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.output = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # Compute gradient of softmax
        # grad = softmax * (grad_output - sum(grad_output * softmax))
        grad_input = self.output * (
            grad_output
            - np.sum(grad_output * self.output, axis=self.dim, keepdims=True)
        )
        return grad_input


def softmax(x, dim=-1):
    """
    Applies the softmax function along a dimension.

    Args:
        x (Variable): Input tensor
        dim (int): Dimension along which to compute softmax

    Returns:
        Variable: Output tensor with softmax applied
    """
    return Softmax(dim)(x)
