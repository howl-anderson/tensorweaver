import numpy as np

from tensorweaver.autodiff.tensor import Tensor
from tensorweaver.operators.faltten import flatten


def test_flatten_2d():
    # Test flattening a 2D tensor_create
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    result = flatten(x)
    expected = np.array([1, 2, 3, 4, 5, 6])
    assert np.array_equal(result.data, expected)


def test_flatten_3d():
    # Test flattening a 3D tensor_create
    x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    result = flatten(x)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    assert np.array_equal(result.data, expected)


def test_flatten_with_start_dim():
    # Test flattening with start_dim=1
    x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    result = flatten(x, start_dim=1)
    expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert np.array_equal(result.data, expected)


def test_flatten_with_end_dim():
    # Test flattening with end_dim=1
    x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    result = flatten(x, end_dim=1)
    expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert np.array_equal(result.data, expected)


def test_flatten_with_start_and_end_dim():
    # Test flattening with both start_dim and end_dim
    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    result = flatten(x, start_dim=1, end_dim=2)
    expected = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    assert np.array_equal(result.data, expected)


def test_flatten_1d():
    # Test flattening a 1D tensor_create (should remain unchanged)
    x = Tensor(np.array([1, 2, 3, 4]))
    result = flatten(x)
    expected = np.array([1, 2, 3, 4])
    assert np.array_equal(result.data, expected)


def test_flatten_negative_end_dim():
    # Test flattening with negative end_dim
    x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    result = flatten(x, start_dim=1, end_dim=-1)
    expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert np.array_equal(result.data, expected)
