import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.reshape import reshape


def test_reshape():
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    y = reshape(x, (6,))

    y.backward()

    assert y.shape == (6,)
    assert np.array_equal(y.data, np.asarray([1, 2, 3, 4, 5, 6]))
    assert x.grad.shape == (2, 3)
    assert np.array_equal(x.grad.data, np.ones((2, 3)))
