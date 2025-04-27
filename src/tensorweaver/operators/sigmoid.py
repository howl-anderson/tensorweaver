import numpy as np
from tensorweaver.autodiff.operator import Operator


class Sigmoid(Operator):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y  # Save output value for backpropagation
        return y

    def backward(self, gy):
        y = self.y
        return gy * y * (1 - y)  # Sigmoid derivative is y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)
