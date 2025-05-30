import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.dropout import dropout


def test_basic_dropout():
    # Test basic dropout functionality
    x = Tensor(np.ones((100, 100)))
    out = dropout(x, p=0.5)

    # Check output shape
    assert out.data.shape == x.data.shape

    # Check if the dropout ratio is close to expected
    zeros = np.sum(out.data == 0)
    total = np.prod(x.data.shape)
    dropout_rate = zeros / total
    assert 0.45 <= dropout_rate <= 0.55  # Allow 5% error margin


def test_different_dropout_rates():
    # Test different dropout rates
    x = Tensor(np.ones((1000,)))
    rates = [0.1, 0.3, 0.5, 0.7, 0.9]

    for p in rates:
        out = dropout(x, p=p)
        zeros = np.sum(out.data == 0)
        dropout_rate = zeros / len(x.data)
        # Allow 5% error margin
        assert p - 0.05 <= dropout_rate <= p + 0.05


def test_training_eval_modes():
    # Test training and evaluation modes
    x = Tensor(np.ones((100,)))

    # Training mode
    train_out = dropout(x, p=0.5, training=True)
    assert not np.array_equal(train_out.data, x.data)

    # Evaluation mode
    eval_out = dropout(x, p=0.5, training=False)
    assert_array_equal(eval_out.data, x.data)


def test_gradient_flow():
    # Test gradient flow
    x = Tensor(np.ones((10,)))
    out = dropout(x, p=0.5)

    # Backward propagation
    out.backward(np.ones_like(out.data))

    # Check gradient
    assert x.grad is not None
    assert x.grad.shape == x.data.shape

    # Check gradient in evaluation mode
    x = Tensor(np.ones((10,)))
    out = dropout(x, p=0.5, training=False)
    out.backward(np.ones_like(out.data))
    assert_array_equal(x.grad, np.ones_like(x.data))


def test_numerical_stability():
    # Test numerical stability
    x = Tensor(np.random.randn(1000) * 1000)  # Use large numbers
    out = dropout(x, p=0.5)

    # Check for infinities or NaNs
    assert not np.any(np.isnan(out.data))
    assert not np.any(np.isinf(out.data))

    # Test small numbers
    x = Tensor(np.random.randn(1000) * 1e-6)
    out = dropout(x, p=0.5)
    assert not np.any(np.isnan(out.data))
    assert not np.any(np.isinf(out.data))


def test_shape_preservation():
    # Test inputs with various shapes
    shapes = [(1,), (10,), (10, 10), (2, 3, 4), (2, 3, 4, 5)]

    for shape in shapes:
        x = Tensor(np.ones(shape))
        out = dropout(x, p=0.5)
        assert out.data.shape == shape


def test_expectation_preservation():
    # Test whether the expected value is preserved
    x = Tensor(np.ones((10000,)) * 2.0)  # Array of all 2s
    out = dropout(x, p=0.5)

    # Due to scaling, non-zero elements should be 4.0
    non_zero_mean = np.mean(out.data[out.data != 0])
    assert 3.8 <= non_zero_mean <= 4.2  # Allow some random error

    # Overall mean should be close to the original value
    mean = np.mean(out.data)
    assert 1.8 <= mean <= 2.2  # Allow some random error


def test_invalid_dropout_rate():
    # Test invalid dropout rates
    x = Tensor(np.ones((10,)))

    with pytest.raises(ValueError):
        dropout(x, p=1.5)  # p > 1

    with pytest.raises(ValueError):
        dropout(x, p=-0.5)  # p < 0
