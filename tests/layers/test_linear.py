import numpy as np

from tensorweaver.autodiff.variable import Variable
from tensorweaver.layers.linear import Linear


def test_liner():
    l = Linear(32, 64)

    x = Variable(np.ones((256, 32)))

    y = l(x)

    assert y.shape == (256, 64)
