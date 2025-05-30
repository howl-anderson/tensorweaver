from tensorweaver.autodiff.operator import Operator


class Equal(Operator):
    def forward(self, x1, x2):
        return x1 == x2

    def backward(self, grad):
        # Equal comparison operation has no meaningful gradient
        return None, None


def eq(x1, x2):
    """Compare whether two tensors are equal.

    Args:
        x1 (Variable): First tensor.
        x2 (Variable): Second tensor.

    Returns:
        Variable: Variable object containing boolean values indicating whether elements at corresponding positions are equal.
    """
    return Equal()(x1, x2)
