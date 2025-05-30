from tensorweaver.autodiff.operator import Operator


class Mul(Operator):
    def forward(self, a, b):
        return a * b

    def backward(self, gy):
        a, b = self.input_data

        return b * gy, a * gy


def mul(x, y):
    return Mul()(x, y)
