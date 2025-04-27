import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.mul import mul


def test_mul():
    a = Tensor(np.asarray(2))
    b = Tensor(np.asarray(3))

    y = mul(a, b)

    y.backward()

    assert np.array_equal(y.data, np.asarray(6))
    assert np.array_equal(a.grad.data, np.asarray(3))
    assert np.array_equal(b.grad.data, np.asarray(2))
