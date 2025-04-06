import numpy as np
from numpy.typing import NDArray

from tensorweaver.autodiff.function import Function
from tensorweaver.autodiff.variable import Variable


class Sin(Function):
    def forward(self, x: NDArray) -> NDArray:
        return np.sin(x)

    def backward(self, x: Variable) -> Variable:
        # lazy import to avoid import circle
        from tensorweaver.operators.cos import cos

        return cos(x)


def sin(x):
    return Sin()(x)
