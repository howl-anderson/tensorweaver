import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.relu import relu


def test_relu_forward_scalar():
    """Test ReLU forward pass with scalar inputs"""
    # Test positive input
    x = Tensor(2.0)
    y = relu(x)
    assert y.data == 2.0

    # Test negative input
    x = Tensor(-2.0)
    y = relu(x)
    assert y.data == 0.0

    # Test zero input
    x = Tensor(0.0)
    y = relu(x)
    assert y.data == 0.0


def test_relu_forward_array():
    """Test ReLU forward pass with array inputs"""
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    y = relu(x)

    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.array_equal(y.data, expected)


def test_relu_backward_scalar():
    """Test ReLU backward pass with scalar inputs"""
    # Test gradient when input > 0
    x = Tensor(2.0)
    y = relu(x)
    y.backward()
    assert x.grad == 1.0  # Gradient should be 1 for positive input

    # Test gradient when input < 0
    x = Tensor(-2.0)
    y = relu(x)
    y.backward()
    assert x.grad == 0.0  # Gradient should be 0 for negative input

    # Test gradient at x = 0
    x = Tensor(0.0)
    y = relu(x)
    y.backward()
    assert x.grad == 0.0  # Gradient at 0 should be 0


def test_relu_backward_array():
    """Test ReLU backward pass with array inputs"""
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    y = relu(x)
    y.backward()

    expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.array_equal(x.grad, expected_grad)


def test_relu_2d_array():
    """Test ReLU with 2D arrays"""
    x_data = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    x = Tensor(x_data)
    y = relu(x)

    # Test forward pass
    expected = np.array([[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
    assert np.array_equal(y.data, expected)

    # Test backward pass with non-uniform gradient
    upstream_grad = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y.backward(upstream_grad)

    expected_grad = np.array([[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
    assert np.array_equal(x.grad, expected_grad)


def test_relu_chain():
    """Test ReLU in a chain of operations"""
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    h = relu(x)  # First ReLU
    y = relu(h)  # Second ReLU

    # Test forward pass
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.array_equal(y.data, expected)

    # Test backward pass
    y.backward()
    expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.array_equal(x.grad, expected_grad)
