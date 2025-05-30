import numpy as np
from tensorweaver.autodiff.operator import Operator


class Tanh(Operator):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        # Derivative of tanh(x) is 1 - tanh^2(x)
        return grad_output * (1 - self.output * self.output)


def tanh(x):
    """
    Applies the hyperbolic tangent function element-wise.

    Args:
        x (Variable): Input tensor

    Returns:
        Variable: Output tensor with tanh applied
    """
    return Tanh()(x)
