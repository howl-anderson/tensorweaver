import numpy as np
from numpy.typing import NDArray

from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class Sin(Operator):
    def forward(self, x: NDArray) -> NDArray:
        return np.sin(x)

    def backward(self, x: Tensor) -> Tensor:
        # lazy import to avoid import circle
        from tensorweaver.operators.cos import cos

        return cos(x)


def sin(x):
    return Sin()(x)
