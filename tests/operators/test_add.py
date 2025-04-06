import numpy as np

from tensorweaver.autodiff.variable import Variable
from tensorweaver.operators.add import add


def test_add():
    a = Variable(np.asarray([1, 2]))
    b = Variable(np.asarray([3, 4]))

    y = add(a, b)

    y.backward()

    assert np.array_equal(y.data, np.asarray([4, 6]))
    assert np.array_equal(a.grad.data, np.ones((2,)))
    assert np.array_equal(b.grad.data, np.ones((2,)))
