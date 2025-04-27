import numpy as np
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.neg import neg


def test_neg():
    x = Tensor(np.asarray(1))

    y = neg(x)

    y.backward()

    np.testing.assert_almost_equal(y.data, -1)
    np.testing.assert_almost_equal(x.grad, -1)
