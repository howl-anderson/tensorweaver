import numpy as np
from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.sigmoid import sigmoid


def test_sigmoid():
    x = Tensor(np.asarray(1))

    y = sigmoid(x)

    y.backward()

    np.testing.assert_almost_equal(y.data, 0.731058578630074)
    np.testing.assert_almost_equal(x.grad, 0.19661193324144993)
