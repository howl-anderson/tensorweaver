import numpy as np
from tensorweaver.autodiff.operator import Operator


class GetItem(Operator):
    def __init__(self, key):
        super().__init__()

        self.key = key
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x[self.key]

    def backward(self, grad):
        # Create a zero array with the same shape as the input
        grad_input = np.zeros(self.input_shape, dtype=grad.dtype)
        # Fill the gradient into the corresponding position
        grad_input[self.key] = grad
        return grad_input


def getitem(x, key):
    """Implements slice operations for tensors.

    Args:
        x (Variable): Input tensor
        key: Slice index, can be an integer, slice object, or tuple

    Returns:
        Variable: Sliced tensor
    """
    return GetItem(key)(x)
