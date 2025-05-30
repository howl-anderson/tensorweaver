import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class Mean(Operator):
    def __init__(self, axis=None, keepdims=False):
        """Mean operator that computes the mean value over specified dimensions.

        Args:
            axis (int or tuple of ints, optional): Dimensions to reduce. If None, reduces all dimensions.
            keepdims (bool, optional): Whether to keep the reduced dimensions with length 1.
        """
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.input_shape = None
        self.n_elements = None

    def forward(self, x):
        self.input_shape = x.shape

        # Calculate number of elements we're averaging over
        if self.axis is None:
            self.n_elements = x.size
        else:
            self.n_elements = np.prod(
                [
                    self.input_shape[i]
                    for i in (
                        self.axis if isinstance(self.axis, tuple) else (self.axis,)
                    )
                ]
            )

        return np.mean(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        """Backward pass for mean operation.

        The gradient for mean operation is the input gradient divided by
        the number of elements we averaged over, broadcast back to the input shape.
        """
        if not self.keepdims and self.axis is not None:
            # If dimensions were squeezed, need to unsqueeze gradient
            grad = np.expand_dims(grad, axis=self.axis)

        # Broadcast gradient and divide by number of elements
        grad_broadcasted = np.broadcast_to(grad, self.input_shape)
        return np.divide(grad_broadcasted, self.n_elements)


def mean(x, axis=None, keepdims=False):
    """Compute the mean of a tensor_create over specified dimensions.

    Args:
        x (Tensor): Input tensor_create.
        axis (int or tuple of ints, optional): Dimensions to reduce. If None, reduces all dimensions.
        keepdims (bool, optional): Whether to keep the reduced dimensions with length 1.

    Returns:
        Tensor: The mean value over specified dimensions.
    """
    return Mean(axis=axis, keepdims=keepdims)(x)
