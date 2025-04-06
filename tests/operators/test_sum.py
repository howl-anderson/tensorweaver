import numpy as np

from tensorweaver.autodiff.variable import Variable
from tensorweaver.operators.sum import sum


def test_sum():
    x = Variable(np.asarray([[1, 2, 3], [4, 5, 6]]))

    y = sum(x)

    y.backward()

    assert np.array_equal(y.data, np.asarray(21))
    assert np.array_equal(x.grad.data, np.ones([2, 3]))
