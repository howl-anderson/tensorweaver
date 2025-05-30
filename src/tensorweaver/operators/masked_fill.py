from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class MaskedFill(Operator):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.mask = None

    def forward(self, x, mask):
        """
        Args:
            x: input tensor
            mask: boolean mask tensor
        Returns:
            output: tensor with elements replaced by value where mask is True
        """
        # Convert mask to numpy array if it's a Variable
        if isinstance(mask, Tensor):
            mask = mask.data

        # Convert mask to boolean array
        mask = mask.astype(bool)

        # Store mask for backward pass
        self.mask = mask

        # Create output array
        result = x.copy()
        result[mask] = self.value
        return result

    def backward(self, grad_output):
        """
        Backward pass for masked_fill.
        The gradient is zero where the mask is True.
        """
        grad_input = grad_output.copy()
        grad_input[self.mask] = 0
        return grad_input, None  # None for mask gradient


def masked_fill(x, mask, value):
    """
    Fills elements of input tensor with value where mask is True.

    Args:
        x (Tensor): input tensor
        mask (Tensor): boolean mask
        value (float): value to fill with

    Returns:
        Tensor: output tensor with masked fill applied
    """
    return MaskedFill(value)(x, mask)
