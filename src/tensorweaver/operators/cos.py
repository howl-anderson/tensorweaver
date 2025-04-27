import numpy as np
from numpy.typing import NDArray

from tensorweaver.autodiff.operator import Operator


class Cos(Operator):
    def forward(self, x: NDArray) -> NDArray:
        return np.cos(x)

    def backward(self, x: NDArray) -> NDArray:
        # lazy import to avoid import circle
        from tensorweaver.operators.sin import sin

        return -1 * sin(x)


def cos(x):
    return Cos()(x)
