import numpy as np
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.exp import exp


def test_exp():
    x = Tensor(np.asarray(1))

    y = exp(x)

    y.backward()

    np.testing.assert_almost_equal(y.data, 2.71828183)
    np.testing.assert_almost_equal(x.grad, 2.71828183)
