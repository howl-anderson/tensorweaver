import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.max import Max


def test_max_scalar():
    """Test max operation with scalar input"""
    x = Tensor(5.0)
    y = Max()(x)

    # Test forward pass
    assert y.data == 5.0

    # Test backward pass
    y.backward()
    assert x.grad == 1.0


def test_max_1d_array():
    """Test max operation with 1D array"""
    x = Tensor(np.array([1.0, 3.0, 2.0, 5.0, 4.0]))
    y = Max()(x)

    # Test forward pass
    assert y.data == 5.0

    # Test backward pass
    y.backward()
    expected_grad = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0]
    )  # Only position of max has gradient
    assert np.array_equal(x.grad, expected_grad)


def test_max_2d_array_global():
    """Test max operation with 2D array (global max)"""
    x = Tensor(np.array([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]]))
    y = Max()(x)

    # Test forward pass
    assert y.data == 6.0

    # Test backward pass
    y.backward()
    expected_grad = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.array_equal(x.grad, expected_grad)


def test_max_2d_array_axis():
    """Test max operation with 2D array along specific axis"""
    x = Tensor(np.array([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]]))

    # Test max along axis 0 (column-wise)
    y0 = Max(axis=0)(x)
    expected0 = np.array([5.0, 4.0, 6.0])
    assert np.array_equal(y0.data, expected0)

    # Test backward pass for axis 0
    y0.backward(np.array([1.0, 1.0, 1.0]))  # Gradient matches output shape
    expected_grad0 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.array_equal(x.grad, expected_grad0)

    # Reset gradient
    x.clean_grad()

    # Test max along axis 1 (row-wise)
    y1 = Max(axis=1)(x)
    expected1 = np.array([3.0, 6.0])
    assert np.array_equal(y1.data, expected1)

    # Test backward pass for axis 1
    y1.backward(np.array([1.0, 1.0]))  # Gradient matches output shape
    expected_grad1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.array_equal(x.grad, expected_grad1)


def test_max_duplicate_values():
    """Test max operation when there are multiple maximum values"""
    x = Tensor(np.array([1.0, 5.0, 2.0, 5.0, 3.0]))
    y = Max()(x)

    # Test forward pass
    assert y.data == 5.0

    # Test backward pass
    y.backward()
    # Only the first occurrence of max should get the gradient
    expected_grad = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(x.grad, expected_grad)


def test_max_3d_array():
    """Test max operation with 3D array"""
    x_data = np.arange(24).reshape(2, 3, 4)
    x = Tensor(x_data)

    # Test global max
    y = Max()(x)
    assert y.data == 23

    # Test max along axis 0
    y0 = Max(axis=0)(x)
    expected0 = np.maximum(x_data[0], x_data[1])
    assert np.array_equal(y0.data, expected0)

    # Test max along axis 1
    y1 = Max(axis=1)(x)
    expected1 = np.maximum.reduce(x_data, axis=1)
    assert np.array_equal(y1.data, expected1)

    # Test max along axis 2
    y2 = Max(axis=2)(x)
    expected2 = np.maximum.reduce(x_data, axis=2)
    assert np.array_equal(y2.data, expected2)
