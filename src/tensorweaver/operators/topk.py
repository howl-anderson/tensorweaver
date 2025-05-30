import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class TopK(Operator):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.indices = None

    def forward(self, x):
        """
        Args:
            x: input tensor
        Returns:
            values: top k values
            indices: indices of top k values
        """
        # Get the last dimension size
        last_dim = x.shape[-1]
        k = min(self.k, last_dim)

        # Partition array to find top k elements
        indices = np.argpartition(x, -k, axis=-1)[..., -k:]

        # Get values for sorting
        values_for_sort = np.take_along_axis(x, indices, axis=-1)

        # Sort the top k elements
        sort_idx = np.argsort(-values_for_sort, axis=-1)
        indices_sorted = np.take_along_axis(indices, sort_idx, axis=-1)

        self.indices = indices_sorted

        # Get the values using advanced indexing
        values = np.take_along_axis(x, indices_sorted, axis=-1)

        return values, indices_sorted

    def backward(self, grad_values, grad_indices=None):
        """
        Backward pass for topk.
        Only values are differentiable, indices are not.
        """
        x = self.input_data[0]
        # Initialize gradient with zeros
        grad_input = np.zeros_like(x)

        # Use saved indices to place gradients
        np.put_along_axis(grad_input, self.indices, grad_values, axis=-1)

        return grad_input


def topk(x, k):
    """
    Returns the k largest elements of the input tensor along the last dimension.

    Args:
        x (Tensor): input tensor
        k (int): number of top elements to return

    Returns:
        tuple: (values, indices) where values are the top k elements and indices are their positions
    """
    return TopK(k)(x)
