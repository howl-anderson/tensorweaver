import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.tril import tril


def test_tril_basic():
    # Test basic 2x2 matrix
    input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input = Tensor(input_data)

    # Default diagonal (k=0)
    output = tril(input)
    expected = np.array([[1, 0], [3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)


def test_tril_different_diagonals():
    # Test with different diagonal values
    input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input = Tensor(input_data)

    # Main diagonal (k=0)
    output = tril(input, diagonal=0)
    expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Above main diagonal (k=1)
    output = tril(input, diagonal=1)
    expected = np.array([[1, 2, 0], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Below main diagonal (k=-1)
    output = tril(input, diagonal=-1)
    expected = np.array([[0, 0, 0], [4, 0, 0], [7, 8, 0]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)


def test_tril_backward():
    # Test gradient computation
    input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input = Tensor(input_data)

    # Forward pass
    output = tril(input)

    # Backward pass with gradient of ones
    grad_output = np.ones_like(input_data)
    output.backward(grad_output)

    # Gradient should be lower triangular
    expected_grad = np.array([[1, 0], [1, 1]], dtype=np.float32)
    np.testing.assert_array_equal(input.grad, expected_grad)


def test_tril_rectangular():
    # Test rectangular matrices
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2x3 matrix
    input = Tensor(input_data)

    # Main diagonal (k=0)
    output = tril(input)
    expected = np.array([[1, 0, 0], [4, 5, 0]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)

    # Above main diagonal (k=1)
    output = tril(input, diagonal=1)
    expected = np.array([[1, 2, 0], [4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)


def test_tril_3d_input():
    # Test 3D input (batch of matrices)
    input_data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32
    )  # 2x2x2 tensor_create
    input = Tensor(input_data)

    output = tril(input)
    expected = np.array([[[1, 0], [3, 4]], [[5, 0], [7, 8]]], dtype=np.float32)
    np.testing.assert_array_equal(output.data, expected)
