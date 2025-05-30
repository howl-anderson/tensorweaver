import numpy as np
from tensorweaver.autodiff.operator import Operator


class Max(Operator):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = np.max(x, axis=self.axis)
        return y

    def backward(self, g):
        x = self.input_data[0]
        mask = np.zeros_like(x)

        if self.axis is None:
            # Global maximum
            idx = np.unravel_index(np.argmax(x), x.shape)
            mask[idx] = g
        else:
            # Maximum along specific axis
            if not isinstance(g, np.ndarray):
                g = np.array(g)

            # Get indices of maximum values along the specific axis
            max_vals = np.max(x, axis=self.axis, keepdims=True)

            # Create a boolean mask for maximum positions
            mask_bool = x == max_vals

            # Reshape gradient to match the reduced shape with keepdims=True
            g = np.expand_dims(g, axis=self.axis)

            # Apply gradient where mask is True
            mask = mask_bool * g

        return mask


def max(x, axis=None):
    return Max(axis)(x)
