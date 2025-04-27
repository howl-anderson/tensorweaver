import numpy as np
from tensorweaver.autodiff.operator import Operator


class Exp(Operator):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input_data[0]
        return gy * np.exp(x)


def exp(x):
    return Exp()(x)
