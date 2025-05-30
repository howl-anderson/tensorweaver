import numpy as np
import pytest

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.transpose import transpose


def test_transpose_2d():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = transpose(x, 0, 1)

    # Test with default gradient (ones)
    y.backward()
    assert np.array_equal(y.data, np.asarray([[1, 4], [2, 5], [3, 6]]))
    assert np.array_equal(x.grad, np.ones([2, 3]))

    # For second test, create a fresh computation graph
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # Recreate x
    y = transpose(x, 0, 1)  # Recreate y with the new x

    # Test with custom gradient
    upstream_grad = np.array([[2.0, 3.0], [0.5, 1.5], [1.0, 2.0]])
    y.backward(upstream_grad)
    expected_grad = upstream_grad.T  # Gradient should be transposed back
    assert np.array_equal(x.grad, expected_grad)


def test_transpose_3d():
    x = Tensor(np.arange(24).reshape(2, 3, 4))
    y = transpose(x, 0, 2)

    # Create non-uniform upstream gradient
    upstream_grad = np.arange(24).reshape(4, 3, 2)
    y.backward(upstream_grad)

    # Expected gradient should be the appropriate transpose of upstream_grad
    expected_grad = np.transpose(upstream_grad, (2, 1, 0))
    assert np.array_equal(x.grad, expected_grad)

    # Check forward pass is still correct
    expected = np.transpose(np.arange(24).reshape(2, 3, 4), (2, 1, 0))
    assert np.array_equal(y.data, expected)


def test_transpose_identity():
    x = Tensor([[1, 2], [3, 4]])
    y = transpose(x, 0, 0)

    # Test with custom gradient for identity transpose
    upstream_grad = np.array([[0.5, 1.5], [2.0, 3.0]])
    y.backward(upstream_grad)

    # For identity transpose, gradient should be exactly the upstream gradient
    assert np.array_equal(x.grad, upstream_grad)
    # Forward pass should not change the data
    assert np.array_equal(y.data, x.data)


def test_transpose_invalid_dims():
    x = Tensor([[1, 2], [3, 4]])

    with pytest.raises(IndexError):
        transpose(x, 0, 2)  # dim1 out of range

    with pytest.raises(IndexError):
        transpose(x, -3, 1)  # dim0 out of range


def test_transpose_backprop_chain():
    # Test backpropagation in a chain of operations
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = transpose(x, 0, 1)  # 3x2 matrix
    z = transpose(y, 0, 1)  # Back to 2x3, should be same as x

    # Use non-uniform gradient
    upstream_grad = np.array([[1.0, 0.5, 2.0], [0.3, 1.5, 0.7]])
    z.backward(upstream_grad)

    # After two transposes, gradient should be exactly the upstream gradient
    assert np.array_equal(x.grad, upstream_grad)
    # z should be equal to x after two transposes
    assert np.array_equal(z.data, x.data)
