import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.layers.linear import Linear


def test_liner():
    l = Linear(32, 64)

    x = Tensor(np.ones((256, 32)))

    y = l(x)

    assert y.shape == (256, 64)
