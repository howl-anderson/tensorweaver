from tensorweaver.autodiff.tensor import Tensor
import numpy as np


def zeros(shape):
    return Tensor(np.zeros(shape))
