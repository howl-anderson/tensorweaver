from tensorweaver.autodiff.variable import Variable
import numpy as np


def ones(*size, dtype=None):
    if dtype is None:
        dtype = np.float32
    return Variable(np.ones(size, dtype=dtype))