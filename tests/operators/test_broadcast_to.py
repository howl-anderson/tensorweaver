import numpy as np

from tensorweaver.autodiff.variable import Variable
from tensorweaver.operators.broadcast_to import broadcast_to


def test_broadcast_to():
    x = Variable(np.asarray([[1, 2, 3]]))

    y = broadcast_to(x, (2, 3))

    y.backward()

    assert np.array_equal(y.data, np.asarray([[1, 2, 3], [1, 2, 3]]))
    assert np.array_equal(x.grad, np.asarray([[2, 2, 2]]))
