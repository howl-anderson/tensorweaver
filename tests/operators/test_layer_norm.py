import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.layer_norm import layer_norm


def test_layer_norm_simple_case():
    # Simple test case with known values
    input = Tensor(
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    )  # shape: (1, 2, 2)
    normalized_shape = (2,)
    weight = Tensor(np.array([1.0, 1.0], dtype=np.float32))
    bias = Tensor(np.array([0.0, 0.0], dtype=np.float32))
    eps = 1e-5

    # Expected values:
    # Mean of last dim: [[1.5, 3.5]]
    # Var of last dim: [[0.25, 0.25]]
    # Normalized: [[[-1.0, 1.0], [-1.0, 1.0]]]
    expected_output = np.array([[[-1.0, 1.0], [-1.0, 1.0]]], dtype=np.float32)

    output = layer_norm(input, normalized_shape, weight, bias, eps)
    np.testing.assert_allclose(output.data, expected_output, rtol=1e-4, atol=1e-4)


def test_layer_norm_with_scale_shift():
    # Test with non-trivial weight and bias
    input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    normalized_shape = (2,)
    weight = Tensor(np.array([2.0, 0.5], dtype=np.float32))
    bias = Tensor(np.array([1.0, -1.0], dtype=np.float32))
    eps = 1e-5

    # Expected values:
    # Normalized: [[[-1.0, 1.0], [-1.0, 1.0]]]
    # After scale and shift: [[[-1.0, -0.5], [-1.0, -0.5]]]
    expected_output = np.array([[[-1.0, -0.5], [-1.0, -0.5]]], dtype=np.float32)

    output = layer_norm(input, normalized_shape, weight, bias, eps)
    np.testing.assert_allclose(output.data, expected_output, rtol=1e-4, atol=1e-4)


def test_layer_norm_zero_mean_unit_var():
    # Test if output has approximately zero mean and unit variance
    input = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    normalized_shape = (4,)
    weight = Tensor(np.ones(4, dtype=np.float32))
    bias = Tensor(np.zeros(4, dtype=np.float32))
    eps = 1e-5

    output = layer_norm(input, normalized_shape, weight, bias, eps)

    # Check if mean is close to 0 and variance is close to 1
    output_mean = np.mean(output.data, axis=-1)
    output_var = np.var(output.data, axis=-1)

    np.testing.assert_allclose(output_mean, 0.0, atol=1e-4)
    np.testing.assert_allclose(output_var, 1.0, atol=1e-4)


def test_layer_norm_backward():
    # Test gradient computation with simple case
    input = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
    normalized_shape = (2,)
    weight = Tensor(np.array([1.0, 1.0], dtype=np.float32))
    bias = Tensor(np.array([0.0, 0.0], dtype=np.float32))
    eps = 1e-5

    output = layer_norm(input, normalized_shape, weight, bias, eps)

    # Backpropagate gradient of 1.0
    grad_output = np.ones_like(output.data)
    output.backward(grad_output)

    # The sum of gradients should be close to 0 for normalized input
    grad_input = (
        input.grad
    )  # Access grad directly from the input variable instead of through creator
    np.testing.assert_allclose(np.sum(grad_input), 0.0, atol=1e-4)


def test_layer_norm_different_shapes():
    shapes = [((2, 2), (2,)), ((2, 3, 2), (2,)), ((2, 2, 2, 2), (2,))]

    for input_shape, norm_shape in shapes:
        input_data = np.ones(input_shape, dtype=np.float32)
        input = Tensor(input_data)
        weight = Tensor(np.ones(norm_shape, dtype=np.float32))
        bias = Tensor(np.zeros(norm_shape, dtype=np.float32))
        eps = 1e-5

        output = layer_norm(input, norm_shape, weight, bias, eps)

        # For constant input, output should be zero (after normalization, before bias)
        expected_output = np.zeros_like(input_data)
        np.testing.assert_allclose(output.data, expected_output, atol=1e-4)
