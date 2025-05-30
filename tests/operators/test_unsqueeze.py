import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.unsqueeze import unsqueeze


def test_unsqueeze_basic():
    # Test basic 1D tensor_create
    input_data = np.array([1, 2, 3], dtype=np.float32)
    input = Tensor(input_data)

    # Unsqueeze at beginning (dim=0)
    output = unsqueeze(input, 0)
    expected = np.array([[1, 2, 3]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Unsqueeze at end (dim=1)
    output = unsqueeze(input, 1)
    expected = np.array([[1], [2], [3]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)


def test_unsqueeze_multidim():
    # Test 2D tensor_create
    input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input = Tensor(input_data)

    # Unsqueeze at beginning
    output = unsqueeze(input, 0)
    expected = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Unsqueeze in middle
    output = unsqueeze(input, 1)
    expected = np.array([[[1, 2]], [[3, 4]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Unsqueeze at end
    output = unsqueeze(input, 2)
    expected = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)


def test_unsqueeze_negative_dim():
    # Test negative dimension indices
    input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input = Tensor(input_data)

    # Unsqueeze at dim=-1 (equivalent to end)
    output = unsqueeze(input, -1)
    expected = np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Unsqueeze at dim=-2
    output = unsqueeze(input, -2)
    expected = np.array([[[1, 2]], [[3, 4]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Unsqueeze at dim=-3 (equivalent to beginning)
    output = unsqueeze(input, -3)
    expected = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)


def test_unsqueeze_backward():
    # Test gradient computation
    input_data = np.array([1, 2, 3], dtype=np.float32)
    input = Tensor(input_data)

    # Forward pass
    output = unsqueeze(input, 0)  # shape becomes (1, 3)

    # Backward pass
    grad_output = np.array(
        [[1, 1, 1]], dtype=np.float32
    )  # gradient matching output shape
    output.backward(grad_output)

    # Gradient should maintain the original shape
    expected_grad = np.array([1, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(input.grad, expected_grad)


def test_unsqueeze_scalar():
    # Test unsqueeze on scalar-like tensor_create
    input_data = np.array(5, dtype=np.float32)
    input = Tensor(input_data)

    # Add dimension at index 0
    output = unsqueeze(input, 0)
    expected = np.array([5], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Verify shape
    assert output.data.shape == (1,)
