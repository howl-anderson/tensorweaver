from tensorweaver.autodiff.operator import Operator
import numpy as np


class Matmul(Operator):
    def forward(self, a, b):
        return a @ b

    def backward(self, gy):
        a, b = self.input_data

        if np.isscalar(gy) or (isinstance(gy, np.ndarray) and gy.ndim == 0):
            return gy * b, gy * a

        if a.ndim > 2 or b.ndim > 2:
            ga = gy @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ gy
            return ga, gb

        ga = gy @ b.T
        gb = a.T @ gy
        return ga, gb


def matmul(x, y):
    return Matmul()(x, y)
