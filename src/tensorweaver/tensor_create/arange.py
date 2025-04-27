import numpy as np

from tensorweaver.autodiff.tensor import Tensor


def arange(start, end, step=1):
    return Tensor(np.arange(start, end, step))
