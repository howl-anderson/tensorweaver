import numpy as np
from tensorweaver.autodiff.operator import Operator


class Pow(Operator):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent
        self.input = None

    def forward(self, x):
        self.input = x
        return np.power(x, self.exponent)

    def backward(self, grad_output):
        # Derivative of x^n is n * x^(n-1)
        return grad_output * self.exponent * np.power(self.input, self.exponent - 1)


def pow(x, exponent):
    """
    Computes element-wise power of input tensor.

    Args:
        x (Variable): Input tensor
        exponent (float): Exponent value

    Returns:
        Variable: Output tensor with power operation applied
    """
    return Pow(exponent)(x)
