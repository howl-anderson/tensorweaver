from tensorweaver.autodiff.tensor import Tensor
import numpy as np


def ones(*size, dtype=None):
    if dtype is None:
        dtype = np.float32
    return Tensor(np.ones(size, dtype=dtype))
