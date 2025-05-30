import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class Cat(Operator):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        self.input_shapes = None

    def forward(self, *tensors):
        """
        Args:
            *tensors: sequence of tensors to concatenate
        Returns:
            output: concatenated tensor
        """
        # Store input shapes for backward pass
        self.input_shapes = [t.shape for t in tensors]

        # Concatenate along specified dimension
        return np.concatenate(tensors, axis=self.dim)

    def backward(self, grad_output):
        """
        Backward pass for concatenation.
        Split the gradient according to the original tensor shapes.
        """
        # Calculate split indices
        split_sizes = [shape[self.dim] for shape in self.input_shapes]
        indices = np.cumsum(split_sizes)[:-1]

        # Split gradient
        grads = np.split(grad_output, indices, axis=self.dim)

        return tuple(grads)


def cat(tensors, dim=0):
    """
    Concatenates a sequence of tensors along a dimension.

    Args:
        tensors (sequence of Variables): tensors to concatenate
        dim (int): dimension along which to concatenate

    Returns:
        Tensor: concatenated tensor
    """
    if isinstance(tensors, (list, tuple)):
        return Cat(dim)(*tensors)
    else:
        return Cat(dim)(tensors[0], tensors[1])
