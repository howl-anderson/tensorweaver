import numpy as np
from tensorweaver.autodiff.operator import Operator


class Unsqueeze(Operator):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        # Convert negative dim to positive
        ndim = input.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim + 1
        return np.expand_dims(input, axis=dim)

    def backward(self, grad_output):
        # Convert negative dim to positive
        ndim = grad_output.ndim
        dim = self.dim if self.dim >= 0 else ndim + self.dim
        return np.squeeze(grad_output, axis=dim)


def unsqueeze(input, dim):
    return Unsqueeze(dim)(input)
