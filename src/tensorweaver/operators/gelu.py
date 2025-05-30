import numpy as np
from tensorweaver.autodiff.operator import Operator


class GELU(Operator):
    def forward(self, input):
        # Calculate CDF (Cumulative Distribution Function) part
        inner = np.sqrt(2 / np.pi) * (input + 0.044715 * input**3)
        cdf = 0.5 * (1 + np.tanh(inner))
        self.inner = inner
        self.cdf = cdf
        return input * cdf

    def backward(self, grad_output):
        x = self.input_data[0]
        cdf = self.cdf
        inner = self.inner
        # Computation of GELU derivative
        # d_tanh = 1 - tanh²
        d_tanh = 1 - np.tanh(inner) ** 2
        # Derivative of the inner function: sqrt(2/π) * (1 + 3 * 0.044715 * x²)
        d_inner = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x * x)
        # Complete derivative: cdf + x * 0.5 * d_tanh * d_inner
        return grad_output * (cdf + x * 0.5 * d_tanh * d_inner)


def gelu(x):
    """Compute the Gaussian Error Linear Unit (GELU) activation function.

    GELU(x) = x * Φ(x) where Φ is the standard normal CDF.

    We use the approximation GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))).

    Args:
        x (Variable): Input tensor

    Returns:
        Variable: Output tensor with GELU activation applied
    """
    f = GELU()
    return f(x)


def gelu_grad(x, gy):
    inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    cdf = 0.5 * (1 + np.tanh(inner))

    # Computation of GELU derivative
    d_tanh = 1 - np.tanh(inner) ** 2
    # Derivative of the inner function: sqrt(2/π) * (1 + 3 * 0.044715 * x²)
    d_inner = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)

    # Complete derivative: cdf + x * d_tanh * d_inner
    return gy * (cdf + x * 0.5 * d_tanh * d_inner)
