from tensorweaver.autodiff.function import Function


class Sub(Function):
    def forward(self, a, b):
        return a - b

    def backward(self, gy):
        return gy, -1 * gy


def sub(a, b):
    return Sub()(a, b)
