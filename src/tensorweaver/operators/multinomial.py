import numpy as np
from tensorweaver.autodiff.operator import Operator
from tensorweaver.autodiff.tensor import Tensor


class Multinomial(Operator):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, probs):
        """
        Args:
            probs: probability distribution tensor
        Returns:
            output: tensor of sampled indices
        """
        # Get the shape of the input
        batch_size = probs.shape[0] if len(probs.shape) > 1 else 1
        num_classes = probs.shape[-1]

        # Reshape probs if needed
        if len(probs.shape) == 1:
            probs = probs.reshape(1, -1)

        # Generate samples for each batch
        samples = np.zeros((batch_size, self.num_samples), dtype=np.int64)
        for i in range(batch_size):
            samples[i] = np.random.choice(
                num_classes, size=self.num_samples, p=probs[i]
            )

        return samples

    def backward(self, grad_output):
        """
        Backward pass for multinomial sampling.
        Sampling operations are not differentiable.
        """
        return None


def multinomial(probs, num_samples):
    """
    Draws samples from a multinomial probability distribution.

    Args:
        probs (Tensor): probability distribution tensor
        num_samples (int): number of samples to draw

    Returns:
        Tensor: tensor of sampled indices
    """
    return Multinomial(num_samples)(probs)
