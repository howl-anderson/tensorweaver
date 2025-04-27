from tensorweaver.autodiff.operator import Operator


class Transpose(Operator):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        shape = list(range(x.ndim))
        shape[self.dim0] = self.dim1
        shape[self.dim1] = self.dim0
        return x.transpose(shape)

    def backward(self, grad):
        shape = list(range(grad.ndim))
        shape[self.dim0] = self.dim1
        shape[self.dim1] = self.dim0
        return grad.transpose(shape)


def transpose(input, dim0, dim1):
    return Transpose(dim0, dim1)(input)
