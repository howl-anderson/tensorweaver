import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class LogSoftmax(Operator):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_output = None

    def forward(self, x):
        # For numerical stability, subtract the maximum value first
        x_max = x.max(axis=self.dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = exp_x.sum(axis=self.dim, keepdims=True)

        # Save softmax output for backpropagation
        self.softmax_output = exp_x / sum_exp_x

        # log_softmax = log(softmax) = x - x_max - log(sum(exp(x - x_max)))
        return (x - x_max) - np.log(sum_exp_x)

    def backward(self, grad_output):
        """
        Gradient computation formula for log_softmax:
        d(log_softmax(x)_i)/d(x_j) = delta_ij - softmax(x)_j
        where delta_ij is the Kronecker function (1 when i=j, 0 otherwise)
        """
        # grad_output has the same shape as the forward output
        # softmax_output is already calculated and saved in forward
        return grad_output - (
            self.softmax_output * grad_output.sum(axis=self.dim, keepdims=True)
        )


def log_softmax(input, dim=-1):
    """Calculate log_softmax of the input tensor.

    log_softmax(x)_i = log(exp(x_i) / sum_j(exp(x_j)))
                     = x_i - log(sum_j(exp(x_j)))

    For numerical stability, the calculation first subtracts the maximum value:
    log_softmax(x)_i = (x_i - max_j(x_j)) - log(sum_j(exp(x_j - max_j(x_j))))

    Args:
        input (Tensor): Input tensor
        dim (int, optional): Dimension to compute softmax over. Defaults to -1.

    Returns:
        Tensor: Result of log_softmax
    """
    return LogSoftmax(dim)(input)
