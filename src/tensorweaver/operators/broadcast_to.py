import numpy as np

from tensorweaver.autodiff.operator import Operator


class BroadcastTo(Operator):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape

        return np.broadcast_to(x, self.target_shape)

    def backward(self, gy):
        # lazy load to avoid import circle
        from tensorweaver.operators.sum_to import np_sum_to

        return np_sum_to(gy, self.original_shape)


def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)
