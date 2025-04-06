import numpy as np

from tensorweaver.autodiff.variable import Variable
from tensorweaver.operators.mul import mul


def test_mul():
    a = Variable(np.asarray(2))
    b = Variable(np.asarray(3))

    y = mul(a, b)

    y.backward()

    assert np.array_equal(y.data, np.asarray(6))
    assert np.array_equal(a.grad.data, np.asarray(3))
    assert np.array_equal(b.grad.data, np.asarray(2))
