import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class CrossEntropy(Operator):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.input = None
        self.target = None
        self.mask = None

    def forward(self, input, target):
        """
        Args:
            input: (N, C) tensor of logits
            target: (N,) tensor of target indices
        Returns:
            scalar loss value
        """
        self.input = input
        # Convert target to numpy array if it's a Tensor
        if isinstance(target, Tensor):
            target = target.data
        self.target = target.astype(np.int64)  # Ensure target is integer type

        # Create mask for ignored indices
        self.mask = self.target != self.ignore_index

        # Compute log softmax
        max_val = np.max(input, axis=1, keepdims=True)
        exp_x = np.exp(input - max_val)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        log_probs = input - max_val - np.log(sum_exp_x)

        # Gather log probabilities of target classes
        n_valid = np.sum(self.mask)
        if n_valid == 0:
            return np.array(0.0)

        batch_size = input.shape[0]
        gathered_logits = log_probs[np.arange(batch_size), self.target]
        gathered_logits = gathered_logits[self.mask]

        return -np.mean(gathered_logits)

    def backward(self, grad_output):
        """
        Compute gradients for cross entropy loss.
        """
        batch_size = self.input.shape[0]
        n_classes = self.input.shape[1]

        # Compute softmax probabilities
        max_val = np.max(self.input, axis=1, keepdims=True)
        exp_x = np.exp(self.input - max_val)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Create one-hot encoding of targets
        target_one_hot = np.zeros_like(probs)
        valid_indices = np.where(self.mask)[0]
        target_one_hot[valid_indices, self.target[valid_indices]] = 1

        # Compute gradient
        n_valid = np.sum(self.mask)
        if n_valid == 0:
            return np.zeros_like(self.input), None

        grad_input = (probs - target_one_hot) / n_valid
        grad_input *= grad_output

        return grad_input, None


def cross_entropy(input, target, ignore_index=-1):
    """
    Computes cross entropy loss between input logits and target indices.

    Args:
        input (Variable): (N, C) tensor of logits
        target (Variable): (N,) tensor of target indices
        ignore_index (int): target indices with this value are ignored

    Returns:
        Variable: scalar loss value
    """
    return CrossEntropy(ignore_index)(input, target)
