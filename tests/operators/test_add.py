import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.add import add


def test_add():
    a = Tensor(np.asarray([1, 2]))
    b = Tensor(np.asarray([3, 4]))

    y = add(a, b)

    y.backward()

    assert np.array_equal(y.data, np.asarray([4, 6]))
    assert np.array_equal(a.grad.data, np.ones((2,)))
    assert np.array_equal(b.grad.data, np.ones((2,)))
