import numpy as np
from tensorweaver.autodiff.operator import Operator


class ReLU(Operator):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, g):
        x = self.input_data[0]
        mask = x > 0
        return g * mask


def relu(x):
    return ReLU()(x)
