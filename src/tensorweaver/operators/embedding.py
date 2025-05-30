from tensorweaver.autodiff.operator import Operator
import numpy as np


class Embedding(Operator):
    def __init__(self, padding_idx=None):
        super().__init__()

        self.padding_idx = padding_idx

    def forward(self, input, weight):
        return weight[input]

    def backward(self, grad_output):
        input, weight = self.input_data
        grad_weight = np.zeros_like(weight)

        # Accumulate gradients to the corresponding embedding vectors for each position
        np.add.at(grad_weight, input, grad_output)

        # If padding_idx is set, the gradient at that position should be zero
        if self.padding_idx is not None:
            grad_weight[self.padding_idx] = 0

        return np.ones_like(input), grad_weight


def embedding(input, weight, padding_idx=None):
    return Embedding(padding_idx)(input, weight)
