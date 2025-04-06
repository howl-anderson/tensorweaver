import numpy as np
from tensorweaver.autodiff.variable import Variable
from tensorweaver.operators.neg import neg


def test_neg():
    x = Variable(np.asarray(1))

    y = neg(x)

    y.backward()

    np.testing.assert_almost_equal(y.data, -1)
    np.testing.assert_almost_equal(x.grad, -1)
