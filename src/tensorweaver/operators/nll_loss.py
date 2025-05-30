import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class NLLLoss(Operator):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.target = None
        self.n_classes = None

    def forward(self, x, target):
        """Calculate negative log likelihood loss.

        Args:
            x: Log probabilities of shape (N, C), typically output from log_softmax
            target: Target class indices of shape (N,)

        Returns:
            Loss value, returns average loss if reduction='mean',
            returns sum of losses if reduction='sum'
        """
        self.target = target
        self.n_classes = x.shape[1]

        # Get negative log probability for each sample's target class
        batch_size = x.shape[0]
        loss = -x[np.arange(batch_size), target]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def backward(self, grad_output):
        """Calculate gradients.

        Gradient calculation rules:
        - For target class i, gradient is -1/N (if reduction='mean')
        - For non-target classes, gradient is 0
        """
        batch_size = self.target.shape[0]
        grad = np.zeros((batch_size, self.n_classes))

        if self.reduction == "mean":
            grad[np.arange(batch_size), self.target] = -1.0 / batch_size
        elif self.reduction == "sum":
            grad[np.arange(batch_size), self.target] = -1.0
        else:  # 'none'
            grad[np.arange(batch_size), self.target] = -1.0
            grad = grad * grad_output.reshape(-1, 1)

        return grad


def nll_loss(input, target, reduction="mean"):
    """Calculate negative log likelihood loss.

    Args:
        input (Tensor): Log probabilities of shape (N, C)
        target (Tensor): Target class indices of shape (N,)
        reduction (str): 'none' | 'mean' | 'sum'

    Returns:
        Tensor: Loss value
    """
    return NLLLoss(reduction)(input, target)
