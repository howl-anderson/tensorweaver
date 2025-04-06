from tensorweaver.autodiff.variable import Variable
import numpy as np


def zeros(shape):
    return Variable(np.zeros(shape))