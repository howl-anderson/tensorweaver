from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class Flatten(Operator):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.input_shape = None

    def forward(self, x):
        # Save input shape for backpropagation
        self.input_shape = x.shape

        # Process negative end_dim
        end_dim = self.end_dim
        if end_dim < 0:
            end_dim = len(x.shape) + end_dim

        # Calculate the product of dimensions to be flattened
        flatten_dims = 1
        for dim in range(self.start_dim, end_dim + 1):
            flatten_dims *= x.shape[dim]

        # Construct new shape
        new_shape = list(x.shape[: self.start_dim])
        new_shape.append(flatten_dims)
        new_shape.extend(x.shape[end_dim + 1 :])

        return x.reshape(tuple(new_shape))

    def backward(self, grad_output):
        # For backpropagation, just reshape the gradient to the original input shape
        return grad_output.reshape(self.input_shape)


def flatten(input, start_dim=0, end_dim=-1):
    """Flatten the tensor over a specified dimension range.

    Args:
        input (Tensor): Input tensor
        start_dim (int, optional): Starting dimension to flatten. Defaults to 0.
        end_dim (int, optional): Ending dimension to flatten (inclusive). Defaults to -1.

    Returns:
        Tensor: Flattened tensor
    """
    return Flatten(start_dim, end_dim)(input)
