import numpy as np
from tensorweaver.autodiff.operator import Operator


class Power(Operator):
    def forward(self, a, b):
        y = a**b
        self.a = a
        self.b = b
        self.y = y
        return y

    def backward(self, grad):
        """Backward pass for power operation.

        For y = a^b:
        dy/da = b * a^(b-1)
        dy/db = a^b * ln(a)
        """
        a, b = self.a, self.b

        # Handle the case where a is 0
        if np.any(a == 0):
            if np.any(b <= 0):
                raise ValueError("0^x is undefined for x <= 0")
            if np.any(b < 1):
                raise ValueError("Derivative of 0^x is undefined for 0 < x < 1")

        # Compute gradients
        ga = b * (a ** (b - 1)) * grad
        gb = (a**b) * np.log(np.where(a > 0, a, 1)) * grad

        return ga, gb


def power(x, y):
    return Power()(x, y)
