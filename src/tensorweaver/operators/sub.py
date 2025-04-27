from tensorweaver.autodiff.operator import Operator


class Sub(Operator):
    def forward(self, a, b):
        return a - b

    def backward(self, gy):
        return gy, -1 * gy


def sub(a, b):
    return Sub()(a, b)
