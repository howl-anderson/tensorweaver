import numpy as np
from numpy.testing import assert_allclose

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.gelu import gelu


def test_basic_gelu():
    # Test basic GELU functionality
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = gelu(x)

    # GELU value at 0 should be 0
    assert_allclose(out.data[2], 0.0, atol=1e-7)

    # GELU is not a standard odd function, but is approximately symmetric around 0
    assert out.data[1] < 0 and out.data[3] > 0
    assert abs(out.data[1]) < abs(out.data[3])  # Negative values are suppressed


def test_gelu_accuracy():
    # Test GELU numerical accuracy
    x = Tensor(np.array([0.0, 1.0, -1.0]))
    out = gelu(x)

    # These are exact values from pre-calculation
    expected = np.array([0.0, 0.841192, -0.158808])
    assert_allclose(out.data, expected, rtol=1e-6)


def test_gelu_gradient():
    # Test gradient calculation
    x = Tensor(np.array([1.0]))
    out = gelu(x)
    out.backward()

    # Gradient at x=1
    assert x.grad[0] > 1.0  # Gradient should be greater than 1

    # Test gradient at x=0
    x = Tensor(np.array([0.0]))
    out = gelu(x)
    out.backward()
    # Gradient at x=0 should be 0.5
    assert_allclose(x.grad, [0.5], rtol=1e-7)


def test_gelu_shape_preservation():
    # Test various input shapes
    shapes = [(1,), (5,), (2, 3), (2, 3, 4)]

    for shape in shapes:
        x = Tensor(np.random.randn(*shape))
        out = gelu(x)
        assert out.data.shape == shape


def test_gelu_numerical_stability():
    # Test numerical stability
    # Test large input values
    x = Tensor(np.array([1000.0, -1000.0]))
    out = gelu(x)
    assert not np.any(np.isnan(out.data))
    assert not np.any(np.isinf(out.data))
    # For large positive numbers, GELU should be close to the input value
    assert_allclose(out.data[0] / x.data[0], 1.0, rtol=1e-3)
    # For large negative numbers, GELU should be close to 0
    assert_allclose(out.data[1], 0.0, atol=1e-7)

    # Test small input values
    x = Tensor(np.array([1e-6, -1e-6]))
    out = gelu(x)
    assert not np.any(np.isnan(out.data))
    assert not np.any(np.isinf(out.data))


def test_gelu_properties():
    # Test basic properties of GELU
    x = Tensor(np.linspace(-5, 5, 1000))
    out = gelu(x)

    # GELU should be positive in the positive region
    positive_mask = x.data > 0
    assert np.all(out.data[positive_mask] > 0)

    # GELU should be negative in the negative region
    negative_mask = x.data < 0
    assert np.all(out.data[negative_mask] < 0)

    # GELU should be close to the input value when the value is large and positive
    large_positive = x.data > 3
    assert np.all(out.data[large_positive] / x.data[large_positive] > 0.95)

    # GELU should be close to 0 when the value is large and negative
    large_negative = x.data < -3
    assert np.all(np.abs(out.data[large_negative]) < np.abs(x.data[large_negative]))


def test_gelu_batch_gradient():
    # Test gradient calculation for batch data
    x = Tensor(np.random.randn(10, 5))
    out = gelu(x)
    out.backward(np.ones_like(out.data))

    # Check gradient shape
    assert x.grad.shape == x.data.shape

    # Gradients should not contain nan or inf
    assert not np.any(np.isnan(x.grad))
    assert not np.any(np.isinf(x.grad))
