from tensorweaver.autodiff.operator import Operator

from tensorweaver.operators.sum_to import np_sum_to


class Add(Operator):
    def forward(self, a, b):
        self.shape_a = a.shape
        self.shape_b = b.shape

        return a + b

    def backward(self, gy):
        ga, gb = gy, gy
        if self.shape_a != self.shape_b:
            ga = np_sum_to(ga, self.shape_a)
            gb = np_sum_to(gb, self.shape_b)

        return ga, gb


def add(x, y):
    return Add()(x, y)
