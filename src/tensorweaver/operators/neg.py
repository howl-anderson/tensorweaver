from tensorweaver.autodiff.operator import Operator


class Neg(Operator):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -1 * gy


def neg(x):
    return Neg()(x)
