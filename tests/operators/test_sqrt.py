import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.sqrt import sqrt


def test_sqrt():
    a = Tensor(3.0)

    b = sqrt(a)

    b.backward()

    assert np.array_equal(b.data, np.asarray(9.0))
    assert np.array_equal(a.grad.data, np.asarray(6.0))
