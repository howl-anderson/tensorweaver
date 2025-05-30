from tensorweaver.autodiff.operator import Operator


class Lt(Operator):
    def forward(self, x1, x2):
        """
        Args:
            x1: first input tensor
            x2: second input tensor
        Returns:
            output: boolean tensor where True if x1 < x2
        """
        return x1 < x2

    def backward(self, grad_output):
        """
        Backward pass for less than operation.
        Since comparison operators are not differentiable,
        we return None for both inputs.
        """
        return None, None


def lt(x1, x2):
    """
    Element-wise less than comparison.

    Args:
        x1 (Variable): first input tensor
        x2 (Variable): second input tensor

    Returns:
        Variable: boolean tensor where True if x1 < x2
    """
    return Lt()(x1, x2)
