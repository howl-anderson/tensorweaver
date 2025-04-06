from tensorweaver.autodiff.function import Function


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -1 * gy


def neg(x):
    return Neg()(x)
