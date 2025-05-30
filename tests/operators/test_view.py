import numpy as np
import pytest

from tensorweaver.operators.view import view
from tensorweaver.autodiff.tensor import Tensor


def test_view_basic():
    # Test basic shape transformation
    input_data = np.array([[1, 2, 3, 4]], dtype=np.float32)  # shape: (1, 4)
    input = Tensor(input_data)

    # Reshape to (2, 2)
    output = view(input, 2, 2)
    expected = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)
    assert output.data.shape == (2, 2)


def test_view_flatten():
    # Test flattening a tensor
    input_data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32
    )  # shape: (2, 2, 2)
    input = Tensor(input_data)

    # Flatten to 1D
    output = view(input, -1)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)
    assert output.data.shape == (8,)


def test_view_with_negative_dim():
    # Test using -1 to infer dimension
    input_data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    input = Tensor(input_data)

    # Reshape to (2, 3) using -1
    output = view(input, 2, -1)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)
    assert output.data.shape == (2, 3)

    # Reshape to (3, 2) using -1
    output = view(input, -1, 2)
    expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)
    assert output.data.shape == (3, 2)


def test_view_backward():
    # Test gradient computation
    input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)  # shape: (2, 2)
    input = Tensor(input_data)

    # Forward pass: reshape to (4,)
    output = view(input, -1)

    # Backward pass
    grad_output = np.array([1, 2, 3, 4], dtype=np.float32)
    output.backward(grad_output)

    # Gradient should maintain the original shape
    expected_grad = np.array([[1, 2], [3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(input.grad, expected_grad)


def test_view_preserve_contiguity():
    # Test that view preserves memory layout
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # shape: (2, 3)
    input = Tensor(input_data)

    # Reshape to (3, 2)
    output = view(input, 3, 2)

    # Check values and shape
    expected = input_data.reshape(3, 2)
    np.testing.assert_array_equal(output.data, expected)
    assert output.data.shape == (3, 2)


def test_view_invalid_size():
    # Test that invalid shapes raise errors
    input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)  # shape: (2, 2)
    input = Tensor(input_data)

    # Try to reshape to incompatible shape
    with pytest.raises(ValueError):
        view(input, 3, 3)  # Should raise error: can't reshape (2,2) to (3,3)
