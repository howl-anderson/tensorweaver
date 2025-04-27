from tensorweaver.autodiff.operator import Operator


class Div(Operator):
    def forward(self, a, b):
        return a / b

    def backward(self, gy):
        a, b = self.input_data
        # Note: Integers to negative integer powers are not allowed.
        return 1 / b * gy, a * -1 / (b**2) * gy


def div(x, y):
    return Div()(x, y)
