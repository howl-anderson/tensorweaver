import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class Sum(Operator):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        if self.dim is None:
            # If no dimension specified, compute the sum of all elements
            result = np.sum(x)
            if self.keepdim:
                result = np.array([result])
        else:
            # Compute sum along the specified dimension
            result = np.sum(x, axis=self.dim, keepdims=self.keepdim)
        return result

    def backward(self, grad):
        # If we need to restore dimensions during backpropagation
        if not self.keepdim and self.dim is not None:
            grad = np.expand_dims(grad, self.dim)

        # Broadcast gradient to input shape
        return np.broadcast_to(grad, self.input_shape)


def sum(x, dim=None, keepdim=False):
    """Calculate the sum of a tensor along the specified dimension.

    Args:
        x (Tensor): Input tensor.
        dim (int, optional): Dimension to reduce. Default is None, meaning compute the sum of all elements.
        keepdim (bool, optional): Whether to keep the output with the same dimensions as the input. Default is False.

    Returns:
        Tensor: Tensor after summation.
    """
    return Sum(dim=dim, keepdim=keepdim)(x)
