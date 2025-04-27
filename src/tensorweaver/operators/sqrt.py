from tensorweaver.autodiff.operator import Operator


class Sqrt(Operator):
    def forward(self, a):
        return a * a

    def backward(self, gy):
        a = self.input_data[0]
        return 2 * a * gy


def sqrt(x):
    return Sqrt()(x)
