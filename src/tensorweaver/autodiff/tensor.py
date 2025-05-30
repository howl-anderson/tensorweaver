# Pytorch reference: https://pytorch.org/docs/stable/tensors.html#torch.Tensor

import numpy as np

from tensorweaver.autodiff.helpers import as_ndarray, as_tensor


class Tensor:
    def __init__(self, data, name=None):
        self.data = as_ndarray(data) if data is not None else None
        self._grad = None
        self.creator = None

        self.name = name

    def backward(self, gradient=None):
        """
        Compute gradients through the computational graph.
        For scalar outputs (e.g. loss values), gradient can be omitted.
        """

        # lazy import for avoid circle import
        from tensorweaver.autodiff.topological_sort import topological_sort

        if gradient is None:
            if self.data.size == 1:  # For scalar outputs
                self.grad = np.ones_like(self.data)
            else:
                self.grad = np.ones_like(self.data)
        else:
            # check if gradient is compatible with self.data
            if isinstance(gradient, Tensor):
                gradient = gradient.data
            if gradient.shape != self.data.shape:
                raise ValueError(
                    f"Gradient shape {gradient.shape} does not match data shape {self.data.shape}"
                )

            self.grad = gradient

        sorted_variables = topological_sort(self)

        for i, var in enumerate(sorted_variables):
            if var.creator is None:
                continue

            back_input = [output.grad for output in var.creator.outputs]
            input_grads = var.creator.backward(*back_input)

            # Clean up gradients we no longer need
            for output in var.creator.outputs:
                if output != self:  # Don't clean the final output gradient
                    output.clean_grad()

            # if input_grads is not list-like, turn to list
            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]

            for input_var, input_grad in zip(var.creator.inputs, input_grads):
                if input_var.grad is None:
                    input_var.grad = input_grad
                else:  # already have grad
                    # Handle batch dimensions by summing over them if needed
                    if input_grad.ndim > input_var.grad.ndim:
                        # Sum over extra dimensions (batch dimensions)
                        axes = tuple(range(input_grad.ndim - input_var.grad.ndim))
                        # Use in-place sum to avoid creating new array
                        np.add.reduce(input_grad, axis=axes, out=input_var.grad)
                    elif input_grad.shape != input_var.grad.shape:
                        raise ValueError(
                            f"Cannot accumulate gradients with different shapes after batch reduction: existing {input_var.grad.shape} vs new {input_grad.shape}"
                        )
                    else:
                        # Use in-place addition
                        np.add(input_var.grad, input_grad, out=input_var.grad)

        # Clean up computational graph after backward pass is complete
        for var in sorted_variables:
            if var.creator is not None:
                var.creator.outputs = None
                var.creator.inputs = None
                var.creator = None

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def clean_grad(self):
        """Clean up gradient and force memory deallocation."""
        if self._grad is not None:
            # Explicitly delete the gradient array
            del self._grad
        self._grad = None

        # We don't clean up computational graph references here anymore
        # as they might be needed for backward pass

    @property
    def shape(self):
        if self.data is None:
            return None
        else:
            return self.data.shape

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if isinstance(value, Tensor):
            raise TypeError(
                f"grad cannot be a Variable object. Got {type(value)} from {value.creator if value.creator else 'unknown'}"
            )
        self._grad = value

    @property
    def T(self):
        from tensorweaver.operators.permute import permute

        return permute(self, list(range(len(self.data.shape) - 1, -1, -1)))

    def reshape(self, *shape):
        from tensorweaver.operators.reshape import reshape

        return reshape(self, shape)

    def permute(self, *dims):
        from tensorweaver.operators.permute import permute

        return permute(self, dims)

    def max(self, axis=None):
        from tensorweaver.operators.max import max

        return max(self, axis)

    def transpose(self, dim0, dim1):
        from tensorweaver.operators.transpose import transpose

        return transpose(self, dim0, dim1)

    def __add__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.add import add

        return add(self, as_tensor(other, self))

    def __radd__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.add import add

        return add(as_tensor(other, self), self)

    def __sub__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.sub import sub

        return sub(self, as_tensor(other, self))

    def __rsub__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.sub import sub

        return sub(as_tensor(other, self), self)

    def __mul__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.mul import mul

        return mul(self, as_tensor(other, self))

    def __rmul__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.mul import mul

        return mul(as_tensor(other, self), self)

    def __truediv__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.div import div

        return div(self, as_tensor(other, self))

    def __rtruediv__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.div import div

        return div(as_tensor(other, self), self)

    def __pow__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.power import power

        return power(self, as_tensor(other, self))

    def __rpow__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.power import power

        return power(as_tensor(other, self), self)

    def __matmul__(self, other):
        # lazy import to avoid import circle
        from tensorweaver.operators.matmul import matmul

        return matmul(self, as_tensor(other, self))

    def __neg__(self):
        # lazy import to avoid import circle
        from tensorweaver.operators.neg import neg

        return neg(self)

    def item(self):
        """Return scalar value. Valid only when data size is 1."""
        if self.data.size != 1:
            raise ValueError("Can only get item value from tensor of size 1")
        return float(self.data)

    def argmax(self, dim=None, keepdim=False):
        """Return the index of the maximum value in the specified dimension.

        Args:
            dim (int, optional): Dimension to reduce. Default is None, indicating to return the index of the maximum value in the entire tensor.
            keepdim (bool, optional): Whether to keep the output with the same dimensions as the input. Default is False.

        Returns:
            Tensor: A Variable object containing the indices of maximum values.
        """
        # lazy import to avoid import circle
        from tensorweaver.operators.argmax import argmax

        return argmax(self, dim=dim, keepdim=keepdim)

    def eq(self, other):
        """Compare whether two tensors are equal.

        Args:
            other (Tensor): Another tensor to compare with.

        Returns:
            Tensor: Variable object containing boolean values indicating whether elements at corresponding positions are equal.
        """
        # lazy import to avoid import circle
        from tensorweaver.operators.eq import eq

        return eq(self, as_tensor(other, self))

    def view_as(self, other):
        """Reshape the tensor to have the same shape as another tensor.

        Args:
            other (Tensor): Tensor with target shape.

        Returns:
            Tensor: Reshaped tensor.
        """
        # lazy import to avoid import circle
        from tensorweaver.operators.view_as import view_as

        return view_as(self, other)

    def sum(self, dim=None, keepdim=False):
        """Calculate the sum of a tensor along the specified dimension.

        Args:
            dim (int, optional): Dimension to reduce. Default is None, meaning compute the sum of all elements.
            keepdim (bool, optional): Whether to keep the output with the same dimensions as the input. Default is False.

        Returns:
            Tensor: Tensor after summation.
        """
        # lazy import to avoid import circle
        from tensorweaver.operators.sum import sum

        return sum(self, dim=dim, keepdim=keepdim)

    def mean(self, axis=None, keepdims=False):
        """Compute the mean value over specified dimensions.

        Args:
            axis (int or tuple of ints, optional): Dimensions to reduce. If None, reduces all dimensions.
            keepdims (bool, optional): Whether to keep the reduced dimensions with length 1.

        Returns:
            Tensor: The mean value over specified dimensions.
        """
        from tensorweaver.operators.mean import mean

        return mean(self, axis=axis, keepdims=keepdims)

    def size(self, dim=None):
        if dim is None:
            return self.data.shape  # Return shape tuple
        else:
            return self.data.shape[dim]

    def unsqueeze(self, dim):
        from tensorweaver.operators.unsqueeze import unsqueeze

        return unsqueeze(self, dim)

    def view(self, *shape):
        from tensorweaver.operators.view import view

        return view(self, *shape)

    def masked_fill(self, mask, value):
        """Fills elements of input tensor with value where mask is True.

        Args:
            mask (Tensor or ndarray): Boolean mask
            value (float): Value to fill with

        Returns:
            Tensor: Output tensor with masked fill applied
        """
        # lazy import to avoid import circle
        from tensorweaver.operators.masked_fill import masked_fill

        # Ensure value is a scalar
        if isinstance(value, Tensor):
            value = (
                value.data.item() if hasattr(value.data, "item") else float(value.data)
            )
        return masked_fill(self, as_tensor(mask), value)

    def __getitem__(self, key):
        """Support for slice operations.

        Args:
            key: Slice index, can be an integer, slice object, or tuple.

        Returns:
            Tensor: Sliced tensor.
        """
        # lazy import to avoid import circle
        from tensorweaver.operators.getitem import getitem

        return getitem(self, key)

    def __lt__(self, other):
        """Support for < operator."""
        from tensorweaver.operators.lt import lt

        if not isinstance(other, Tensor):
            other = Tensor(other)
        return lt(self, other)

    def tolist(self):
        """Convert the tensor to a Python list."""
        return self.data.tolist()

    def detach(self):
        """Detach the tensor from the computational graph."""
        var = Tensor(self.data)
        var.data = self.data  # reference to the same memory

        return var
