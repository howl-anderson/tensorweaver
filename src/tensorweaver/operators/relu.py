import numpy as np
from tensorweaver.autodiff.function import Function


class Relu(Function):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, g):
        x = self.input_data[0]
        mask = (x > 0)
        return g * mask
    

def relu(x):
    return Relu()(x)