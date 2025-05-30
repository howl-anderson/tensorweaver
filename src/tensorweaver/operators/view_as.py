import numpy as np
from tensorweaver.autodiff.operator import Operator


class ViewAs(Operator):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.target_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


def view_as(x, other):
    """Reshape the tensor to have the same shape as another tensor.

    Args:
        x (Variable): Tensor to be reshaped.
        other (Variable): Tensor with the target shape.

    Returns:
        Variable: Reshaped tensor.
    """
    if hasattr(other, "data"):
        target_shape = other.data.shape
    else:
        target_shape = np.array(other).shape
    return ViewAs(target_shape)(x)
