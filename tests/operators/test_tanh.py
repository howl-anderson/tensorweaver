import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.tanh import tanh


def test_tanh_scalar():
    """Test tanh operation with scalar input"""
    x = Tensor(0.0)
    y = tanh(x)

    # Test forward pass
    assert np.isclose(y.data, 0.0)

    # Test backward pass
    y.backward()
    assert np.isclose(x.grad, 1.0)  # derivative of tanh at 0 is 1


def test_tanh_array():
    """Test tanh operation with array input"""
    x = Tensor(np.array([0.0, 1.0, -1.0]))
    y = tanh(x)

    # Test forward pass
    expected = np.array([0.0, 0.7615941559557649, -0.7615941559557649])
    assert np.allclose(y.data, expected)

    # Test backward pass
    y.backward()
    expected_grad = 1 - np.tanh(x.data) ** 2
    assert np.allclose(x.grad, expected_grad)


def test_tanh_2d_array():
    """Test tanh operation with 2D array"""
    x = Tensor(np.array([[0.0, 1.0], [-1.0, 2.0]]))
    y = tanh(x)

    # Test forward pass
    expected = np.tanh(x.data)
    assert np.allclose(y.data, expected)

    # Test backward pass
    y.backward()
    expected_grad = 1 - np.tanh(x.data) ** 2
    assert np.allclose(x.grad, expected_grad)


def test_tanh_large_values():
    """Test tanh operation with large values"""
    x = Tensor(np.array([100.0, -100.0]))
    y = tanh(x)

    # Test forward pass
    assert np.allclose(y.data, np.array([1.0, -1.0]), atol=1e-7)

    # Test backward pass
    y.backward()
    expected_grad = 1 - np.tanh(x.data) ** 2
    assert np.allclose(x.grad, expected_grad)
