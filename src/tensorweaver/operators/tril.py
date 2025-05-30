import numpy as np
from tensorweaver.autodiff.operator import Operator


class Tril(Operator):
    def __init__(self, diagonal=0):
        super().__init__()
        self.diagonal = diagonal

    def forward(self, input):
        return np.tril(input, k=self.diagonal)

    def backward(self, grad_output):
        return np.tril(grad_output, k=self.diagonal)


def tril(input, diagonal=0):
    return Tril(diagonal)(input)
