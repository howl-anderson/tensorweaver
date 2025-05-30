import numpy as np
from tensorweaver.autodiff.operator import Operator


class Dropout(Operator):
    def __init__(self, p=0.5):
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError("Dropout rate must be in range [0, 1)")
        self.p = p

    def forward(self, input):
        if self.training:
            self.mask = np.random.rand(*input.shape) > self.p
            return (
                input * self.mask / (1 - self.p)
            )  # Scale to keep the expected value unchanged
        return input

    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask / (1 - self.p)
        return grad_output


def dropout(input, p=0.5, training=True):
    dropout_layer = Dropout(p)
    dropout_layer.training = training
    return dropout_layer(input)
