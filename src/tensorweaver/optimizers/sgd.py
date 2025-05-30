from tensorweaver.optimizers.optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params)

        self.learning_rate = lr

    def update_one_parameter(self, p):
        """Update a single parameter.

        Args:
            p (Variable): Parameter to update
        """
        if p.grad is None:
            return

        # If the gradient has more dimensions than the parameter,
        # we need to sum over the extra dimensions
        if p.grad.ndim > p.data.ndim:
            axes = tuple(range(p.grad.ndim - p.data.ndim))
            # Use in-place reduction to avoid creating new array
            grad = np.add.reduce(p.grad, axis=axes)
        else:
            grad = p.grad

        # Update the parameter in-place
        np.subtract(p.data, self.learning_rate * grad, out=p.data)

        # Clean up gradients and force deallocation
        p.clean_grad()
        del grad  # Explicitly delete the temporary gradient
