import numpy as np

from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.helpers import as_tensor


class Reshape(Operator):
    def __init__(self, target_shape):
        super().__init__()

        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape

        return np.reshape(x, self.target_shape)

    def backward(self, gy):
        return np.reshape(gy, self.original_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_tensor(x)
    else:
        return Reshape(shape)(x)
