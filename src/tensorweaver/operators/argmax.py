import numpy as np
from tensorweaver.autodiff.operator import Operator


class Argmax(Operator):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            # If no dimension specified, return the index of the maximum value in the entire tensor
            result = np.argmax(x)
            if self.keepdim:
                result = np.array([result])
        else:
            # Calculate the indices of maximum values along the specified dimension
            result = np.argmax(x, axis=self.dim)
            if self.keepdim:
                # Insert a new axis in the reduced dimension
                result = np.expand_dims(result, axis=self.dim)
        return result

    def backward(self, grad):
        # argmax operation has no meaningful gradient
        return None


def argmax(x, dim=None, keepdim=False):
    """Returns the indices of the maximum values along the specified dimension.

    Args:
        x (Variable): Input tensor.
        dim (int, optional): Dimension to reduce. Default is None, indicating to return the index of the maximum value in the entire tensor.
        keepdim (bool, optional): Whether to keep the output with the same dimensions as the input. Default is False.

    Returns:
        Variable: A Variable object containing the indices of maximum values.
    """
    return Argmax(dim=dim, keepdim=keepdim)(x)
