import numpy as np
from numpy.testing import assert_allclose

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.softmax import softmax


def test_basic_softmax():
    # Test basic softmax calculation
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    out = softmax(x)
    expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
    assert_allclose(out.data, expected, rtol=1e-6)


def test_softmax_numerical_stability():
    # Test numerical stability - large numbers
    x = Tensor(np.array([[1000.0, 2000.0, 3000.0]]))
    out = softmax(x)
    # Maximum value should be close to 1, others close to 0
    assert_allclose(out.data[0, -1], 1.0, rtol=1e-6)
    assert np.all(out.data[0, :-1] < 1e-10)

    # Test numerical stability - small numbers
    x = Tensor(np.array([[-1000.0, -2000.0, -3000.0]]))
    out = softmax(x)
    assert_allclose(out.data[0, 0], 1.0, rtol=1e-6)
    assert np.all(out.data[0, 1:] < 1e-10)


def test_softmax_edge_cases():
    # Test zero vector
    x = Tensor(np.array([[0.0, 0.0, 0.0]]))
    out = softmax(x)
    expected = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert_allclose(out.data, expected, rtol=1e-6)

    # Test vector with identical values
    x = Tensor(np.array([[2.0, 2.0, 2.0]]))
    out = softmax(x)
    expected = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert_allclose(out.data, expected, rtol=1e-6)


def test_softmax_shape():
    # Test inputs with different shapes
    x = Tensor(np.random.randn(2, 3))
    out = softmax(x)
    assert out.data.shape == (2, 3)
    # Verify that each row sums to 1
    assert_allclose(np.sum(out.data, axis=1), np.array([1.0, 1.0]), rtol=1e-6)


def test_softmax_gradient():
    # Test gradient calculation
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    out = softmax(x)
    # Use a non-uniform upstream gradient
    grad_output_val = np.array([[0.1, 0.2, 0.7]])
    out.backward(grad_output_val)

    # Verify gradient shape
    assert x.grad.shape == x.data.shape

    # Verify gradient correctness
    softmax_output = out.data
    # Result calculated according to the backward formula
    grad_sum = np.sum(grad_output_val * softmax_output, axis=-1, keepdims=True)
    expected_grad = softmax_output * (grad_output_val - grad_sum)

    print(f"Actual grad: {x.grad}")
    print(f"Expected grad: {expected_grad}")

    assert_allclose(x.grad, expected_grad, rtol=1e-6, atol=1e-7)
