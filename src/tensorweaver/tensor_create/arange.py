import numpy as np

from tensorweaver.autodiff.variable import Variable


def arange(start, end, step=1):
    return Variable(np.arange(start, end, step))
